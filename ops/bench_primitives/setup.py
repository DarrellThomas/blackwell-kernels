# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PRIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'csrc', 'primitives', 'linalg'))
COMMON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'csrc', 'common'))

setup(
    name='bench_primitives',
    ext_modules=[
        CUDAExtension(
            'bench_primitives',
            sources=[
                os.path.join(PRIM_DIR, 'syrk_f32_sm120.cu'),
                os.path.join(PRIM_DIR, 'trmm_f32_sm120.cu'),
                'csrc/bindings.cu',
            ],
            include_dirs=[COMMON_DIR],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', '--use_fast_math', '--expt-relaxed-constexpr',
                    '-lineinfo', '-arch=sm_120a',
                ],
            },
            libraries=['cublas'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
