#!/usr/bin/env bash
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
# Build blackwell_kernels PyTorch extension
# Uses CUDA 13.2 toolkit via /usr/local/cuda-13 symlink.
set -euo pipefail

export CUDA_HOME=/usr/local/cuda-13
export PATH=/usr/local/cuda-13/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0a"

cd "$(dirname "$0")"
rm -rf build/

echo "Building with $(nvcc --version | grep release)..."
python3 setup.py build_ext --inplace
echo "Done. Extension at: python/blackwell_kernels/_C*.so"
