#!/bin/bash
# This script creates a config YAML file
# List all the config files
layers=1
neurons=10
scripts=$(ls config_layers_"$layers"_neurons_"$neurons"/*.yaml)

# Create the config file
echo "$scripts";

for script in $scripts; do
    python ea-nn-ik-solver-pytorch-6DoF.py --config-path "$script" &
done


# Print a success message
echo "successfully running experiments based created yaml files!"