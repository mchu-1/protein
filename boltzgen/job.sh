#!/bin/bash

# Ensure script stops on first error
set -e

echo "Starting BoltzGen Run..."

boltzgen run 3di3-config.yaml \
  --output output/3di3 \
  --protocol nanobody-anything \
  --num_designs 50 \
  --budget 10

echo "Run complete."
