#!/bin/bash

# Ensure script stops on first error
set -e

echo "Starting BoltzGen 1K Design Run..."

# Run the Modal app with optimized parallel settings
# 1000 designs split into batches of 50 (20 parallel workers)
# This maximizes throughput to complete within 2h
python generate_nanobodies.py \
  --config 3di3-config.yaml \
  --num_designs 1000 \
  --batch_size 50 \
  --gpu A100

echo "Run complete. Outputs in boltzgen/output/"
