import os
import sys
import math

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
import argparse
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path
import inspect

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train hyperbolic nanogpt with learning rate sweep")
    parser.add_argument("--lr_scale", type=float, default=1.0, help="Global learning rate scaling factor")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="nanogpt-hyperbolic", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/team name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags for wandb")
    return parser.parse_args()

# Configure shared memory limits for Triton kernels
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# Optional import for flex_attention if available
try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False
    print("FlexAttention not available, using standard attention implementation")

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.bfloat16)
        w_f8 = w.div(w_s).to(torch.bfloat16)
        # out = torch._scaled_mm(
        #     x_f8,
        #     w_f8.T,
        #     out_dtype=torch.bfloat16,
        #     scale_a=x.new_tensor(x_s, dtype=torch.float32),
        #     scale_b=x.new_tensor(w_s, dtype=torch.float32),
        #     use_fast_accum=True,
        # )
        out = x_f8 @ w_f8.T
        return out, x_f8, w_f8

    return impl(x, w)


@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.bfloat16), w.to(torch.bfloat16)


@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[
    Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        # grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        # grad_x = torch._scaled_mm(
        #     grad_f8,
        #     w_f8.T.contiguous().T,
        #     out_dtype=torch.bfloat16,
        #     scale_a=grad_inv_s,
        #     scale_b=w_inv_s,
        #     use_fast_accum=False,
        # )
        # # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        # grad_w = torch._scaled_mm(
        #     x_f8.T.contiguous(),
        #     grad_f8.T.contiguous().T,
        #     out_dtype=torch.float32,
        #     scale_a=x_inv_s,
        #     scale_b=grad_inv_s,
        #     use_fast_accum=False,
        # ).T
        grad_bf16 = grad.div(grad_s).to(torch.bfloat16)
        grad_x = grad_bf16 @ w_f8
        grad_w = (x_f8.T @ grad_bf16).float().T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)


@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)


def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None


def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Hyperbolic geometry utility functions

def clamp_curvature(c):
    if isinstance(c, torch.Tensor):
        return torch.clamp(c, min=1e-4, max=1.0)
    else:
        return max(1e-4, min(c, 1.0))

def mobius_addition(x, y, c):
    """Mobius addition in hyperbolic space with curvature c"""
    c = clamp_curvature(c)
    # Compute norms
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    y_norm = torch.norm(y, dim=-1, keepdim=True)
    # Compute the inner product
    inner_product = torch.sum(x * y, dim=-1, keepdim=True)
    
    # Compute numerator and denominator following the standard formula
    numerator = (1 + 2*c * inner_product + c * (y_norm ** 2)) * x + \
                (1 - c * (x_norm ** 2)) * y
    denominator = 1 + 2*c * inner_product + (c ** 2) * (x_norm ** 2) * (y_norm ** 2)
    
    return numerator / denominator

def scaling_factor(x, c):
    """Compute scaling factor for hyperbolic space with curvature c"""
    c = clamp_curvature(c)
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    return 2/(1+c*x_norm**2)

def expmap(x, v, c):
    """Exponential map from tangent space to hyperbolic space with curvature c"""
    c = clamp_curvature(c)
    scaling_factor_x = scaling_factor(x, c)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    second_term = (1/c**0.5)*torch.tanh((c*scaling_factor_x*v_norm**2/2)**0.5)*v/v_norm
    return mobius_addition(x, second_term, c)

def logmap(x, u, c):
    """Logarithmic map from hyperbolic space to tangent space with curvature c"""
    c = clamp_curvature(c)
    scaling_factor_x = scaling_factor(x, c)
    mob_addition = mobius_addition(-x, u, c)
    addition_norm = torch.norm(mob_addition, dim=-1, keepdim=True)
    constant_factor = 2 / (scaling_factor_x * c**0.5)
    direction_factor = mob_addition / addition_norm
    arg = torch.clamp((c * addition_norm) ** 0.5, min=-0.999, max=0.999)  # Single-line fix
    return constant_factor * torch.arctanh(arg) * direction_factor

def calculate_reference_point(x, per_head_curvature=False):
    """Calculate reference point for hyperbolic operations"""
    B, T, C = x.size()
    ref_point = torch.zeros_like(x[:, :1, :])
    if T > 1:
        ref_point = x[:, :-1, :]
        ref_point = F.pad(ref_point, (0, 0, 1, 0), mode='constant', value=0)
    return ref_point

HYPERBOLIC_EPSILON = 1e-4  # Increased from 1e-5 for more aggressive clamping


def mobius_addition(x, y, c):
    """Mobius addition in hyperbolic space with curvature c"""
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(min=0)  # Ensure non-negative
    y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True).clamp(min=0)  # Ensure non-negative
    inner_product = torch.sum(x * y, dim=-1, keepdim=True)

    # c is assumed to be a positive tensor passed in
    c = c.clamp(min=HYPERBOLIC_EPSILON)  # Ensure c is positive

    numerator = (1 + 2 * c * inner_product + c * y_norm_sq) * x + \
                (1 - c * x_norm_sq) * y
    denominator = 1 + 2 * c * inner_product + (c * c) * x_norm_sq * y_norm_sq

    # Prevent division by zero or near-zero
    return numerator / (denominator + HYPERBOLIC_EPSILON)


def scaling_factor(x, c):
    """Compute scaling factor for hyperbolic space with curvature c"""
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(min=0)
    # c is assumed to be a positive tensor passed in
    c = c.clamp(min=HYPERBOLIC_EPSILON)  # Ensure c is positive
    denominator = 1 + c * x_norm_sq
    return 2 / (denominator + HYPERBOLIC_EPSILON)  # Add epsilon here too


def expmap(x, v, c):
    """Exponential map from tangent space to hyperbolic space with curvature c"""
    # c is assumed to be a positive tensor passed in
    c = c.clamp(min=HYPERBOLIC_EPSILON)  # Ensure c is positive
    sqrt_c = c.sqrt()

    v_norm = torch.norm(v, dim=-1, keepdim=True)
    # Prevent division by zero if v_norm is zero
    safe_v_norm = v_norm + HYPERBOLIC_EPSILON

    lambda_x = scaling_factor(x, c)  # Already has epsilon safety

    # Calculate coefficient for v safely
    coeff = (1 / sqrt_c) * torch.tanh(sqrt_c * lambda_x * v_norm / 2)

    second_term = coeff * (v / safe_v_norm)

    # Ensure second_term is numerically stable (e.g., not NaN if v was zero)
    second_term = torch.nan_to_num(second_term, nan=0.0, posinf=0.0, neginf=0.0)

    return mobius_addition(x, second_term, c)


def logmap(x, u, c):
    """Logarithmic map from hyperbolic space to tangent space with curvature c"""
    # c is assumed to be a positive tensor passed in
    c = c.clamp(min=HYPERBOLIC_EPSILON)  # Ensure c is positive
    sqrt_c = c.sqrt()

    mob_add = mobius_addition(-x, u, c)  # Already uses epsilon
    mob_add_norm = torch.norm(mob_add, dim=-1, keepdim=True)

    # Critical: Ensure argument to arctanh is strictly within (-1, 1)
    # Term inside sqrt: c * mob_add_norm^2
    # Need sqrt(c) * mob_add_norm < 1
    # Clamp mob_add_norm to be slightly less than 1/sqrt(c) before calculating arg
    max_norm = (1.0 / sqrt_c) - HYPERBOLIC_EPSILON
    # --- FIX IS HERE ---
    # Clamp min first using a number, then clamp max using the tensor
    safe_mob_add_norm = mob_add_norm.clamp(min=0.0)
    safe_mob_add_norm = safe_mob_add_norm.clamp(max=max_norm)
    # --- END FIX ---

    # Calculate arctanh argument safely
    arg_to_arctanh = sqrt_c * safe_mob_add_norm
    # Clamp again just before arctanh for safety, ensuring it's less than 1
    # Using clamp with only max (Number) is fine
    clamped_arg = arg_to_arctanh.clamp(max=0.9)  # More aggressive clamping below 1.0
    # Ensure positive after sqrt/potential numerical issues before arctanh
    clamped_arg = clamped_arg.clamp(min=0.0)  # Clamp min separately too

    # Compute arctanh
    arctanh_val = torch.arctanh(clamped_arg)

    lambda_x = scaling_factor(x, c)  # Already has epsilon safety

    # Final coefficient calculation with safe denominators
    constant_factor = 2 / (lambda_x * sqrt_c + HYPERBOLIC_EPSILON)
    direction_factor = mob_add / (mob_add_norm + HYPERBOLIC_EPSILON)

    # Handle potential NaN if mob_add_norm was zero -> direction_factor NaN
    direction_factor = torch.nan_to_num(direction_factor, nan=0.0)

    return constant_factor * arctanh_val * direction_factor


# Ensure calculate_reference_point handles the T=1 case cleanly (already does)
def calculate_reference_point(x):
    """Calculate reference point for hyperbolic operations"""
    B, T, C = x.size()
    # Origin is a valid point on the manifold (Poincaré ball)
    ref_point = torch.zeros_like(x[:, :1, :])
    if T > 1:
        # Use previous token's state as the reference point's location
        ref_point = x[:, :-1, :]
        # Pad the start of the sequence with the origin reference point
        ref_point = F.pad(ref_point, (0, 0, 1, 0), mode='constant', value=0)
    return ref_point  # Ensure this is on the same device as x


# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i: base_i + self.world_size]
            update_prev()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)  # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, bias=True, dropout=0.0, 
                 curvature=1.0, map_back_after_attention=True):
        super().__init__()
        assert dim % num_heads == 0
        
        # key, query, value projections
        self.c_attn = nn.Linear(dim, 3 * dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.n_head = num_heads
        self.n_embd = dim
        self.dropout = dropout
        
        # Hyperbolic parameters
        self.c = curvature
        self.map_back_after_attention = map_back_after_attention
        
        # RoPE
        self.rotary = Rotary(dim // num_heads, max_seq_len)
        
        # Flash attention support check
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                 .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: Tensor):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate reference point for hyperbolic operations
        reference_point = calculate_reference_point(x)
        
        # Map input to hyperbolic tangent space
        x_hyperbolic = logmap(reference_point, x, self.c)
        
        # Get query, key, values from linear projection
        qkv = self.c_attn(x_hyperbolic)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply rotary position embeddings
        q = self.rotary(q)
        k = self.rotary(k)
        
        # Perform attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Map back to hyperbolic space if specified
        if self.map_back_after_attention:
            y = expmap(reference_point, y, self.c)
        
        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y, reference_point


class MLP(nn.Module):
    def __init__(self, dim: int, bias=True, dropout=0.0, curvature=1.0, map_back_after_attention=True):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.map_back_after_attention = map_back_after_attention
        self.c = curvature
        
    def forward(self, x, reference_point=None):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
        # Map back to hyperbolic space if not already done by attention
        if not self.map_back_after_attention and reference_point is not None:
            x = expmap(reference_point, x, self.c)
            
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, bias=True, 
                 dropout=0.0, layer_idx: int = 0, curvature_mode='parametric',
                 curvature=1.0, map_back_after_attention=True, per_head_curvature=True):
        super().__init__()

        # Handle curvature - four modes: fixed, parametric, tied, or random
        if curvature_mode == 'fixed':
            self.c = torch.tensor(curvature)
        elif curvature_mode == 'parametric':
            self.c = nn.Parameter(torch.tensor(curvature))
            self.c.requires_grad = True
        elif curvature_mode == 'tied':
            # Use a temporary value that will be replaced with shared parameter
            self.c = 1.0
        else:  # Default to random initialization
            if not per_head_curvature:
                self.c = nn.Parameter(torch.rand(1))  # Single random value for the entire block
                self.c.requires_grad = True
            else:
                self.c = nn.Parameter(torch.rand(num_heads).repeat_interleave(dim//num_heads))
                self.c.requires_grad = True

        # Layer normalization and layers
        #self.ln_1 = LayerNorm(dim, bias=bias)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, bias=bias, dropout=dropout,
                                        curvature=self.c, map_back_after_attention=map_back_after_attention)
        #self.ln_2 = LayerNorm(dim, bias=bias)
        self.mlp = MLP(dim, bias=bias, dropout=dropout,
                       curvature=self.c, map_back_after_attention=map_back_after_attention)

    def forward(self, x):
        attn_output, reference_point = self.attn(norm(x))
        x = x + attn_output
        x = x + self.mlp(norm(x), reference_point)
        return x


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int,
                 bias=True, dropout=0.0, curvature_mode='parametric', curvature=1.0,
                 map_back_after_attention=True, per_head_curvature=True, use_embedding_curvature=False):
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.bias = bias
        self.dropout = dropout
        
        # Hyperbolic parameters
        self.curvature_mode = curvature_mode
        self.curvature = curvature
        self.map_back_after_attention = map_back_after_attention
        self.per_head_curvature = per_head_curvature
        
        # Create shared curvature parameter if using tied mode
        self.shared_curvature = None
        if curvature_mode == 'tied':
            if per_head_curvature:
                # Create one parameter per head, shared across all blocks
                self.shared_curvature = nn.Parameter(torch.rand(n_head).repeat_interleave(n_embd//n_head))
            else:
                # Create a single parameter shared across all blocks
                self.shared_curvature = nn.Parameter(torch.tensor(1.0).view(1))
        
        # Optional embedding curvature
        if use_embedding_curvature:
            self.embedding_curvature = nn.Parameter(torch.tensor(1.0))
        else:
            self.embedding_curvature = None
        
        # Input embedding
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(
                dim=n_embd, 
                num_heads=n_head, 
                max_seq_len=block_size,
                bias=bias,
                dropout=dropout,
                layer_idx=i,
                curvature_mode=curvature_mode,
                curvature=curvature,
                map_back_after_attention=map_back_after_attention,
                per_head_curvature=per_head_curvature
            ) for i in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        
        # Output projection
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Tie weights
        self.transformer.wte.weight = self.lm_head.weight
        
        # Pass the shared curvature parameter if using tied mode
        if curvature_mode == 'tied' and self.shared_curvature is not None:
            for block in self.transformer.h:
                block.c = self.shared_curvature
                block.attn.c = self.shared_curvature
                block.mlp.c = self.shared_curvature
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter) and module.dim() == 0:
            # For scalar parameters like curvature
            torch.nn.init.uniform_(module, a=0.0, b=1.0)
    
    def forward(self, input_seq: torch.Tensor, target_seq: torch.Tensor = None, sliding_window_num_blocks=None):
        device = input_seq.device
        b = 1  # Hardcoded batch size
        t = input_seq.size(0)  # sequence length
        
        if input_seq.ndim == 1:
            input_seq = input_seq.unsqueeze(0)  # Add batch dimension
        
        if target_seq is not None and target_seq.ndim == 1:
            target_seq = target_seq.unsqueeze(0)  # Add batch dimension
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # position indices
        
        # Forward the GPT model
        tok_emb = self.transformer.wte(input_seq)  # token embeddings
        pos_emb = self.transformer.wpe(pos)  # position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Apply embedding curvature if specified
        if self.embedding_curvature is not None:
            reference_point = calculate_reference_point(x)
            x = expmap(reference_point, x, self.embedding_curvature)
        
        # Process through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = norm(x)
        
        # Calculate logits and loss
        logits = self.lm_head(x)
        
        if target_seq is not None:
            # If we have targets, calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1), 
                                   reduction='sum' if self.training else 'mean')
            return loss
        else:
            # For inference, only return the logits
            return logits


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)  # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)  # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64,
                             non_blocking=True)  # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    val_tokens = 10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48 * 1024  # sequence length for training
    val_seq_len = 4 * 64 * 1024  # sequence length for validation
    
    # optimization
    num_iterations = 1770  # number of iterations to run
    cooldown_frac = 0.4  # fraction of training spent cooling down the learning rate
    lr_scale = 1.0  # Global learning rate scaling factor for all optimizer learning rates
    dropout = 0.0  # dropout rate
    
    # architecture
    vocab_size = 50257  # vocabulary size
    block_size = 1024  # maximum sequence length
    n_layer = 12  # number of transformer layers
    n_head = 12  # number of attention heads
    n_embd = 768  # embedding dimension
    bias = True  # whether to use bias in Linear layers
    
    # hyperbolic parameters
    curvature_mode = 'tied'  # 'fixed', 'parametric', 'tied', or 'random'
    curvature = 1.0  # Fixed curvature value when curvature_mode is 'fixed'
    curvature_initialization: list[float] = field(default_factory=lambda: [1.0-1.0e-3] + [1.0e-3] * 11)  # per-layer initialization
    map_back_after_attention = False  # whether to map back to hyperbolic space after attention or after the MLP
    per_head_curvature = True  # whether to use a different curvature for each head
    use_embedding_curvature = False  # whether to use a curvature element also for the embedding layer
    
    # evaluation and logging
    val_loss_every = 125  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
    use_wandb = True  # Whether to use Weights & Biases for logging


args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
# This code now works with any number of GPUs (was originally optimized for 8xH100)
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.

# Update hyperparameters from command line arguments
if master_process:
    cli_args = parse_args()
    for key, value in vars(cli_args).items():
        if hasattr(args, key):
            setattr(args, key, value)

# Initialize wandb if enabled (only on master process)
if master_process and args.use_wandb:
    try:
        import wandb
        # Force CLI login if not already logged in
        try:
            wandb.ensure_login()
            print("WandB login successful")
        except:
            print("Please login to wandb using the CLI prompt")
            import subprocess
            subprocess.run(["wandb", "login"])
        
        # Use entity from arguments or environment variable
        wandb_entity = cli_args.wandb_entity
        if not wandb_entity or wandb_entity == "$(whoami)":
            # Try to get from environment variable or use default
            wandb_entity = os.environ.get("WANDB_ENTITY", "aisparks")
            
        wandb_tags = cli_args.wandb_tags.split(",") if cli_args.wandb_tags else None
        wandb_project = cli_args.wandb_project or os.environ.get("WANDB_PROJECT", "nanogpt-hyperbolic")
        wandb_name = cli_args.wandb_name or os.environ.get("WANDB_NAME", f"run-{uuid.uuid4()}")
        
        print(f"Initializing wandb with project={wandb_project}, entity={wandb_entity}")
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=asdict(args),
            name=wandb_name,
            tags=wandb_tags,
        )
        print("Successfully initialized wandb")
    except ImportError:
        print("Warning: wandb not installed. Proceeding without wandb logging.")
        args.use_wandb = False
    except Exception as e:
        print(f"Error initializing wandb: {str(e)}")
        print("Proceeding without wandb logging")
        args.use_wandb = False

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)


def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)


# begin by printing this file (the Python code)
print0(code)
print0("=" * 100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")


def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


print0(nvidia_smi())
print0("=" * 100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(
    vocab_size=args.vocab_size, 
    block_size=max(args.train_seq_len, args.val_seq_len),
    n_layer=args.n_layer,
    n_head=args.n_head, 
    n_embd=args.n_embd,
    bias=args.bias,
    dropout=args.dropout,
    curvature_mode=args.curvature_mode,
    curvature=args.curvature,
    map_back_after_attention=args.map_back_after_attention,
    per_head_curvature=args.per_head_curvature,
    use_embedding_curvature=args.use_embedding_curvature
).cuda()

for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# Watch model gradients and weights with wandb
if master_process and args.use_wandb:
    wandb.watch(
        model,
        log="all",  # log both gradients and weights
        log_freq=10,  # log every 10 steps
        log_graph=False  # log the model graph
    )

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [dict(params=head_params, lr=args.lr_scale * 0.22), 
               dict(params=embed_params, lr=args.lr_scale * 0.6), 
               dict(params=scalar_params, lr=args.lr_scale * 0.04)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=args.lr_scale * 0.05, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]


# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations  # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1


# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)


def get_window_size_blocks(step: int):
    x = step / args.num_iterations  # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)


model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])  # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for name, param in model.named_parameters():  # Iterate with names for potential debugging
        if param.grad is not None:  # Check if gradient exists before reducing
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                # Get window size blocks if the model is using blockmasks
                window_size_blocks = get_window_size_blocks(step) if hasattr(model, 'create_blockmasks') else None
                val_loss += model(inputs, targets, window_size_blocks)
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
            console=True)
        
        # Log metrics to wandb
        if master_process and args.use_wandb:
            metrics = {
                "val/loss": val_loss.item(),
                "train/time_ms": training_time_ms,
                "train/step_avg_ms": training_time_ms / max(step, 1),
                "train/step": step,
            }
            wandb.log(metrics, step=step)
            
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(),
                       optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # Get window size blocks if the model is using blockmasks
    window_size_blocks = get_window_size_blocks(step) if hasattr(model, 'create_blockmasks') else None
    loss = model(inputs, targets, window_size_blocks)
    loss.backward()
    for name, param in model.named_parameters():  # Iterate with names for potential debugging
        if param.grad is not None:  # Check if gradient exists before reducing
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)  # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    
    # Log training metrics to wandb
    if master_process and args.use_wandb and step % 10 == 0:  # Log every 10 steps to reduce overhead
        lr_adam = optimizer1.param_groups[0]['lr']
        lr_muon = optimizer2.param_groups[0]['lr']
        metrics = {
            "train/loss": loss.item(),
            "train/lr_adam": lr_adam,
            "train/lr_muon": lr_muon,
            "train/lr_scale": args.lr_scale,
        }
        wandb.log(metrics, step=step)
        
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
        console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

# Close wandb if it was initialized
if master_process and args.use_wandb:
    wandb.finish()

dist.destroy_process_group()

# -----------------------------------------------------------------------------
# Utility function for blockmasks (moved from GPT class)

def create_blockmasks(input_seq: Tensor, sliding_window_num_blocks: Tensor):
    # For compatibility with the updated model which doesn't require blockmasks
    # This function can return None if we're not using flex_attention
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except ImportError:
        return None, None
    
    BLOCK_SIZE = 128
    docs = (input_seq == 50256).cumsum(0)

    def document_causal(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[q_idx] == docs[kv_idx]
        return causal_mask & document_mask

    def dense_to_ordered(dense_blockmask: Tensor):
        num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
        return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

    # manual block mask creation by @YouJiacheng
    assert len(input_seq) % BLOCK_SIZE == 0
    NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
    block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
    causal_blockmask_any = block_idx[:, None] >= block_idx
    causal_blockmask_all = block_idx[:, None] > block_idx
    docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
    docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
    document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
    document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
    blockmask_any = causal_blockmask_any & document_blockmask_any
    blockmask_all = causal_blockmask_all & document_blockmask_all
    partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
    full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

    def build_bm(window_size_blocks: Tensor) -> BlockMask:
        return BlockMask.from_kv_blocks(
            torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
            partial_kv_indices,
            torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
            full_kv_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=document_causal,
        )

    # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
    return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)
