#!/usr/bin/env python

from setuptools import find_packages, setup

package_name = "fsbi"
version = "1.0"
exclusions = ["notebooks", "sbi-logs"]

_packages = find_packages(exclude=exclusions)

_base = [
    "torch",
    "numpy",
    "matplotlib",
    "scipy",
    "seaborn",
    "scikit-learn",
    "pyyaml",
    "sbi",
    "tqdm",
        ]

setup(
    name=package_name,
    version=version,
    description="Simulation-based inference for meta-learning plasticity rules",
    author="Basile Confavreux and Poornima Ramesh",
    url="https://github.com/VogelsLab/fSBI",
    packages=_packages,
    install_requires=_base,
)
