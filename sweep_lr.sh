#!/bin/bash

# Define learning rate scales to try
LR_SCALES=(0.005 0.01 0.05 0.1 0.5 1)

# Set WandB environment variables (will be passed to all runs)
export WANDB_PROJECT="nanogpt-hyperbolic-lrs"
export WANDB_ENTITY="aisparks"  # Change this to your organization/username
export WANDB_TAGS="lr_sweep,hyperbolic"

# Number of GPUs to use
NUM_GPUS=1 # Adjust based on your setup

echo "Starting learning rate sweep with ${#LR_SCALES[@]} different scales"
echo "Using WandB entity: $WANDB_ENTITY"
echo "Using WandB project: $WANDB_PROJECT"

# Login to wandb first to ensure credentials are correct
wandb login

for lr_scale in "${LR_SCALES[@]}"; do
  echo "Running with learning rate scale: $lr_scale"
  
  # Set a unique name for this run
  export WANDB_NAME="lr_scale_${lr_scale}"
  
  # Run training with specified learning rate
  torchrun --nproc_per_node=$NUM_GPUS train_gpt.py \
    --lr_scale=$lr_scale \
    --use_wandb
  
  # Optional: wait between runs
  echo "Completed run with lr_scale=$lr_scale"
  sleep 5
done

echo "Learning rate sweep completed!" 