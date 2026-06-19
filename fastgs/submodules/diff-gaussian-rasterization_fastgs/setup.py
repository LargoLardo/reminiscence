#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil


def get_nvcc_ccbin():
    if os.name == 'nt':
        return None
    if os.getenv("CXX"):
        return os.getenv("CXX")
    if os.getenv("CC"):
        return os.getenv("CC")
    for candidate in [
        "x86_64-conda-linux-gnu-g++",
        "x86_64-conda-linux-gnu-gcc",
        "g++-11",
        "g++-10",
        "g++-9",
        "g++-8",
    ]:
        path = shutil.which(candidate)
        if path:
            return path
    return None

ccbin = get_nvcc_ccbin()
nvcc_args = ["-allow-unsupported-compiler"]
if ccbin:
    nvcc_args += ["-ccbin", ccbin]

setup(
    name="diff_gaussian_rasterization_fastgs",
    packages=['diff_gaussian_rasterization_fastgs'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_fastgs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/adam.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            include_dirs=["/usr/include"],
            extra_compile_args={"nvcc": nvcc_args + ["-I/usr/include", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
