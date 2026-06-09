#!/bin/bash

set -e

DATA_ROOT=${1:?Please provide dataset root path}
OUTPUT_ROOT=${2:?Please provide output root path}

SCRIPT_PATH="benchmark_single_scene.py"

# Mip-NeRF scenes indoor
for scene in bonsai counter kitchen room; do
    python "$SCRIPT_PATH" \
        --dataset "$DATA_ROOT/360_v2/" \
        --output_root "$OUTPUT_ROOT/m360/" \
        --scene "$scene" \
        --images images_2 # indoor
done

# Mip-NeRF scenes outdoor
for scene in bicycle flowers garden stump treehill; do
    python "$SCRIPT_PATH" \
        --dataset "$DATA_ROOT/360_v2/" \
        --output_root "$OUTPUT_ROOT/m360/" \
        --scene "$scene" \
        --images images_4 # outdoor
done

# Tanks and Temples scenes
for scene in train truck; do
    python "$SCRIPT_PATH" \
        --dataset "$DATA_ROOT/tandt/" \
        --output_root "$OUTPUT_ROOT/tt/" \
        --scene "$scene" \
        --images images
done

# Deep Blending scenes
for scene in drjohnson playroom; do
    python "$SCRIPT_PATH" \
        --dataset "$DATA_ROOT/db/" \
        --output_root "$OUTPUT_ROOT/db/" \
        --scene "$scene" \
        --images images
done