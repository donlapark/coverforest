import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


if sys.platform == "win32":
    allocator_libs = []  # type: ignore
    extra_compile_args = ["/openmp", "/O2"]
    extra_link_args = ["/openmp"]
elif sys.platform == "darwin":
    allocator_libs = ["jemalloc"]
    extra_compile_args = [
        "-Xpreprocessor",
        "-fopenmp",
        "-O3",
        "-ffast-math",
        "--std=c++17",
    ]
    extra_link_args = ["-lomp"]
else:
    allocator_libs = ["jemalloc"]
    extra_compile_args = [
        "-fopenmp",
        "-O3",
        "-ffast-math",
        "--std=c++17",
    ]
    extra_link_args = ["-fopenmp"]

extension_args = dict(
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)
ext_modules = [
    Extension(
        name="coverforest._giqs",
        sources=["coverforest/_giqs.pyx"],
        **extension_args,
    ),
]

setup(
    entry_points=None,
    ext_modules=cythonize(
        ext_modules,
        annotate=False,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "cdivision": True,
            "legacy_implicit_noexcept": True,
        },
    ),
)
