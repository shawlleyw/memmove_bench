from setuptools import setup, Extension, find_packages
import pybind11
import torch
from pybind11.setup_helpers import build_ext, Pybind11Extension

TORCH_HOME = torch.__path__[0]
TORCH_LIB = f"{TORCH_HOME}/lib"
TORCH_INCLUDES = [f"{TORCH_HOME}/include/torch/csrc/api/include", f"{TORCH_HOME}/include"]

print(f"torch at {TORCH_HOME}")

exts = [
    Pybind11Extension(
        "memmove_c",
        ["csrc/bindings.cc", "csrc/cpp_move.cc"],
        include_dirs=[
            pybind11.get_include(),
            *TORCH_INCLUDES,
            "csrc",
        ],
        library_dirs=[
            TORCH_LIB,
        ],
        extra_compile_args=["-fPIC", "-ltorch"],
        extra_link_args=[
            '-Wl,-rpath,' + TORCH_LIB, "-ltorch"
        ],
        libraries=['torch', 'c10', 'torch_cpu'],
        language="c++",
    ),
]

setup(
    name="memmove",
    version="0.0",
    ext_modules=exts,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(".")
)