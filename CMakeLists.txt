cmake_minimum_required(VERSION 3.24)
project(blackwell_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Common include directory
include_directories(csrc/common)

# BF16 GEMM kernel (standalone, for testing without PyTorch)
add_library(bf16_gemm STATIC csrc/gemm/bf16_gemm_sm120.cu)
target_compile_options(bf16_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -lineinfo>
)

# Flash Attention kernel
add_library(flash_attn STATIC csrc/attention/flash_attn_sm120.cu)
target_compile_options(flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -lineinfo>
)
