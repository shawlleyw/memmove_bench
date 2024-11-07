from setuptools import setup, Extension, find_packages
import pybind11
import torch
from pybind11.setup_helpers import build_ext, Pybind11Extension
from torch.utils import cpp_extension

TORCH_HOME = torch.__path__[0]
TORCH_LIB = f"{TORCH_HOME}/lib"
TORCH_INCLUDES = [f"{TORCH_HOME}/include/torch/csrc/api/include", f"{TORCH_HOME}/include"]

print(f"torch at {TORCH_HOME}")

exts = [
    cpp_extension.CppExtension(
        "memmove_c",
        ["csrc/bindings.cc", "csrc/cpp_move.cc", "csrc/cuda_move.cu"],
        include_dirs=[
            pybind11.get_include(),
            *TORCH_INCLUDES,
            "csrc",
        ],
        extra_compile_args=[
            "-O2",
        ],
        library_dirs=[
            TORCH_LIB,
        ],
        libraries=['torch', 'c10', 'torch_cpu'],
        language="c++",
    ),
]

setup(
    name="memmove",
    version="0.0",
    ext_modules=exts,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=find_packages(".")
)