#!/bin/bash

# Define learning rate scales to try
LR_SCALES=(0.005 0.01 0.05 0.1 0.5 1)

# WandB configuration
WANDB_PROJECT="nanogpt-hyperbolic-lrs"
# Get wandb entity from environment or use default
WANDB_ENTITY=${WANDB_ENTITY:-$(whoami)}  # Use environment variable or username
WANDB_TAGS="lr_sweep,hyperbolic"

# Number of GPUs to use
NUM_GPUS=1 # Adjust based on your setup

echo "Starting learning rate sweep with ${#LR_SCALES[@]} different scales"
echo "Using WandB entity: $WANDB_ENTITY"

for lr_scale in "${LR_SCALES[@]}"; do
  echo "Running with learning rate scale: $lr_scale"
  
  # Run training with specified learning rate
  torchrun --nproc_per_node=$NUM_GPUS train_gpt.py \
    --lr_scale=$lr_scale \
    --use_wandb \
    --wandb_project=$WANDB_PROJECT \
    --wandb_entity=$WANDB_ENTITY \
    --wandb_name="lr_scale_${lr_scale}" \
    --wandb_tags=$WANDB_TAGS
  
  # Optional: wait between runs
  echo "Completed run with lr_scale=$lr_scale"
  sleep 5
done

echo "Learning rate sweep completed!" 