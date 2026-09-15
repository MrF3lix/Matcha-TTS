#!/usr/bin/env python
# All project metadata lives in pyproject.toml.
# This file only exists to build the Cython extension.
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

exts = [
    Extension(
        name="matcha.utils.monotonic_align.core",
        sources=["matcha/utils/monotonic_align/core.pyx"],
    )
]

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(exts, language_level=3),
)
