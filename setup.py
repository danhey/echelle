#!/usr/bin/env python

import os
import sys
from setuptools import setup

setup(
    name="echelle",
    version='1.3',
    author="Daniel Hey",
    url="https://github.com/danhey/echelle",
    packages=["echelle"],
    description="Neat tools for echelle diagrams.",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=["numpy", "matplotlib", "astropy"],
)