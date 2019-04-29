#!/usr/bin/env python

import os
import sys
from setuptools import setup

setup(
    name="echelle",
    version='0.0.0.0.0.8',
    author="Daniel Hey",
    url="https://github.com/danielhey/echelle",
    packages=["echelle"],
    description="Neat tools for echelle diagrams.",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=["numpy", "matplotlib", "bokeh"],
)