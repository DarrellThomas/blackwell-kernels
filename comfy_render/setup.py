# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="blackwell_kernels",
    version="0.1.0",
    description="Custom CUDA kernels optimized for RTX 5090 (sm_120)",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            "blackwell_kernels._C",
            [
                os.path.join(ROOT, "csrc/comfy_render/flash_attn_sm120a.cu"),
                os.path.join(ROOT, "csrc/comfy_render/group_norm_linear_sm120a.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_120a,code=sm_120a",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
            include_dirs=[
                os.path.join(ROOT, "csrc", "common"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
