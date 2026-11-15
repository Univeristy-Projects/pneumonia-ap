#!/bin/bash

# Exit immediately if any training run fails
set -e

echo "========================================"
echo "Starting 5-Seed Training Pipeline (42-46)"
echo "========================================"

for seed in {42..46}; do

    python3 train.py \
        --model_name simple_cnn \
        --seed "$seed"
        
    echo "Finished seed $seed successfully."
done

echo ""
echo "========================================"
echo "All 5 training runs completed!"
echo ""
