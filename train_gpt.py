import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Configure shared memory limits for Triton kernels
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention


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

# def mobius_addition(x, y, c):
#     """Mobius addition in hyperbolic space with curvature c"""
#     # Compute norms
#     x_norm = torch.norm(x, dim=-1, keepdim=True)
#     y_norm = torch.norm(y, dim=-1, keepdim=True)
#     # Compute the inner product
#     inner_product = torch.sum(x * y, dim=-1, keepdim=True)

#     # Ensure c is on the same device as x and with the same dtype
#     c = c.to(x.device).to(x.dtype) if isinstance(c, torch.Tensor) else torch.tensor(c, device=x.device, dtype=x.dtype)

#     # Compute numerator and denominator following the standard formula
#     numerator = (1 + 2*c * inner_product + c * (y_norm ** 2)) * x + \
#                 (1 - c * (x_norm ** 2)) * y
#     denominator = 1 + 2*c * inner_product + (c ** 2) * (x_norm ** 2) * (y_norm ** 2)

#     # Add small epsilon to avoid division by zero
#     epsilon = 1e-6
#     return numerator / (denominator + epsilon)

# def scaling_factor(x, c):
#     """Compute scaling factor for hyperbolic space with curvature c"""
#     x_norm = torch.norm(x, dim=-1, keepdim=True)
#     # Ensure c is on the same device as x and with the same dtype
#     c = c.to(x.device).to(x.dtype) if isinstance(c, torch.Tensor) else torch.tensor(c, device=x.device, dtype=x.dtype)
#     return 2/(1+c*x_norm**2)

# def expmap(x, v, c):
#     """Exponential map from tangent space to hyperbolic space with curvature c"""
#     scaling_factor_x = scaling_factor(x, c)
#     v_norm = torch.norm(v, dim=-1, keepdim=True)

#     # Ensure c is on the same device as x and with the same dtype
#     c = c.to(x.device).to(x.dtype) if isinstance(c, torch.Tensor) else torch.tensor(c, device=x.device, dtype=x.dtype)

#     # Add small epsilon to avoid division by zero
#     epsilon = 1e-6
#     # Safe division with epsilon
#     safe_v_norm = v_norm + epsilon
#     second_term = (1/c**0.5)*torch.tanh((c*scaling_factor_x*v_norm**2/2)**0.5)*v/safe_v_norm
#     return mobius_addition(x, second_term, c)

# def logmap(x, u, c):
#     """Logarithmic map from hyperbolic space to tangent space with curvature c"""
#     # Ensure c is on the same device as x and with the same dtype
#     c = c.to(x.device).to(x.dtype) if isinstance(c, torch.Tensor) else torch.tensor(c, device=x.device, dtype=x.dtype)

#     scaling_factor_x = scaling_factor(x, c)
#     mob_addition = mobius_addition(-x, u, c)
#     addition_norm = torch.norm(mob_addition, dim=-1, keepdim=True)
#     # Add small epsilon to avoid division by zero
#     epsilon = 1e-6
#     constant_factor = 2 / (scaling_factor_x * (c + epsilon)**0.5 + epsilon)
#     arg_sqrt = (c * addition_norm)
#     direction_factor = mob_addition / (addition_norm + epsilon)
#     arg = torch.clamp(arg_sqrt.clamp(min=0), max=1.0 - epsilon)**0.5 # Clamp sqrt input, then clamp result for arctanh
#     # Ensure arctanh input is strictly < 1
#     result = constant_factor * torch.arctanh(arg.clamp(-1.0 + epsilon, 1.0 - epsilon)) * direction_factor
#     return result

# def calculate_reference_point(x):
#     """Calculate reference point for hyperbolic operations"""
#     B, T, C = x.size()
#     ref_point = torch.zeros_like(x[:, :1, :])
#     if T > 1:
#         ref_point = x[:, :-1, :]
#         ref_point = F.pad(ref_point, (0, 0, 1, 0), mode='constant', value=0)
#     return ref_point

HYPERBOLIC_EPSILON = 1e-5  # Increased from 1e-8


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
    clamped_arg = arg_to_arctanh.clamp(max=1.0 - HYPERBOLIC_EPSILON)
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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128, curvature=1.0,
                 map_back_after_attention=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Ensure curvature is a tensor with bfloat16 dtype
        self.c = curvature if isinstance(curvature, torch.Tensor) else torch.tensor(curvature, dtype=torch.bfloat16)
        self.map_back_after_attention = map_back_after_attention
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std  # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"

        # Calculate reference point for hyperbolic operations
        reference_point = calculate_reference_point(x)

        # Ensure curvature is on same device and dtype as input
        c = self.c.to(x.device).to(x.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c,
                                                                                                  device=x.device,
                                                                                                  dtype=x.dtype)

        # Map input to hyperbolic tangent space
        x_hyperbolic = logmap(reference_point, x, c)

        # Project to QKV
        q, k, v = F.linear(x_hyperbolic, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads,
                                                                                        self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)

        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)  # @KoszarskyB & @Grad62304977
        else:  # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v

        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask,
                           scale=0.12).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)  # re-assemble all head outputs side by side

        # Output projection
        y = self.c_proj(y)

        # Map back to hyperbolic space if specified
        if self.map_back_after_attention:
            y = expmap(reference_point, y, c)

        return y, reference_point


class MLP(nn.Module):
    def __init__(self, dim: int, curvature=1.0, map_back_after_attention=True):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977
        # Ensure curvature is a tensor with bfloat16 dtype
        self.c = curvature if isinstance(curvature, torch.Tensor) else torch.tensor(curvature, dtype=torch.bfloat16)
        self.map_back_after_attention = map_back_after_attention

    def forward(self, x: Tensor, reference_point=None):
        x = self.c_fc(x)
        x = F.relu(
            x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)

        # Map back to hyperbolic space if not already done by attention
        if not self.map_back_after_attention and reference_point is not None:
            # Ensure curvature is on same device and dtype as input
            c = self.c.to(x.device).to(x.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c,
                                                                                                      device=x.device,
                                                                                                      dtype=x.dtype)
            x = expmap(reference_point, x, c)

        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int, curvature_mode='parametric',
                 curvature=1.0, map_back_after_attention=True):
        super().__init__()

        # Handle curvature - three modes: fixed, parametric, or random
        if curvature_mode == 'fixed':
            self.c = torch.tensor(curvature)  # Ensure it's a tensor with bfloat16 dtype
        elif curvature_mode == 'parametric':
            self.c = nn.Parameter(torch.tensor(curvature))
        else:  # Default to random initialization
            self.c = nn.Parameter(torch.rand(1))
            self.c.requires_grad = True

        # Pass curvature parameter directly instead of just its value
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, curvature=self.c,
                                        map_back_after_attention=map_back_after_attention) if layer_idx != 7 else None
        self.mlp = MLP(dim, curvature=self.c, map_back_after_attention=map_back_after_attention)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.map_back_after_attention = map_back_after_attention

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        reference_point = None

        if self.attn is not None:
            attn_output, reference_point = self.attn(norm(x), ve, block_mask)
            x = x + attn_output
        else:
            # Calculate reference point and map to hyperbolic tangent space when attention is skipped
            reference_point = calculate_reference_point(x)
            c = self.c.to(x.device).to(x.dtype) if isinstance(self.c, torch.Tensor) else torch.tensor(self.c,
                                                                                                      device=x.device,
                                                                                                      dtype=x.dtype)
            x = logmap(reference_point, x, c)

        mlp_output = self.mlp(norm(x), reference_point)
        x = x + mlp_output

        return x


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int,
                 curvature_mode='parametric', curvature=1.0, map_back_after_attention=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

        # Store hyperbolic parameters for blocks
        self.curvature_mode = curvature_mode
        self.curvature = curvature
        self.map_back_after_attention = map_back_after_attention

        # Initialize blocks with hyperbolic parameters
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, max_seq_len, i,
                  curvature_mode=curvature_mode,
                  curvature=curvature,
                  map_back_after_attention=map_back_after_attention)
            for i in range(num_layers)
        ])

        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=True, x_s=(model_dim ** 0.5) / 448, w_s=24 / 448, grad_s=1 / 448)
        self.lm_head.weight.detach().zero_()  # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
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

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm,
                       short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq,
                               reduction='sum' if self.training else 'mean')
        return loss


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
    train_seq_len = 48 * 1024  # FlexAttention sequence length
    val_seq_len = 4 * 64 * 1024  # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1770  # number of iterations to run
    cooldown_frac = 0.4  # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # hyperbolic parameters
    curvature_mode = 'random'  # 'fixed', 'parametric', or 'random'
    curvature = 1.0  # Fixed curvature value when curvature_mode is 'fixed'
    map_back_after_attention = False  # whether to map back to hyperbolic space after attention or after the MLP
    # evaluation and logging
    val_loss_every = 25  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False


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

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len),
                       curvature_mode=args.curvature_mode,
                       curvature=args.curvature,
                       map_back_after_attention=args.map_back_after_attention).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [dict(params=head_params, lr=0.02), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.01)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=0.022, momentum=0.95, rank=rank, world_size=world_size)
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
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
            console=True)
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
    model(inputs, targets, get_window_size_blocks(step)).backward()
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
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
        console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
