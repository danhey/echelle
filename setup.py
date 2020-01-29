#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/echelle*")
    sys.exit()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

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
    install_requires=install_requires,
)