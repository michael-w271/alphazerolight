
from setuptools import setup, Extension, find_packages
import pybind11
import os

# Define the extension module
ext_modules = [
    Extension(
        "alpha_zero_light.mcts.mcts_cpp",
        ["src/alpha_zero_light/mcts/mcts_cpp.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="alpha_zero_light",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
)
