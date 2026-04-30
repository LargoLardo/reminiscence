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
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ext.cpp"],
            include_dirs=["/usr/include"],
            extra_compile_args={"nvcc": nvcc_args + ["-I/usr/include"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
