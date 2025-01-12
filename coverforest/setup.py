from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import sys


if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'


ext_modules = [
    Extension(
        "_giqs",
        ["_giqs.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(ext_modules, annotate=True)
)
