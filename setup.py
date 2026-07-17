# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name="gfields",
    version="1.0.0",
    author="Jorge Condor",
    description="A Mitsuba 3 extension library for ray tracing volumetric primitives "
                "(Gaussian / Gabor / Epanechnikov kernels), with stochastic "
                "level-of-detail and steerable tomography/PRB integrators, regression scripts and a viewer.",
    long_description=(
        "Gabor-fields renderer built on 'Don't Splat your Gaussians: Volumetric "
        "Ray-Traced Primitives for Modeling and Rendering Scattering and "
        "Emissive Media' (Condor et al., SIGGRAPH 2025). Ships the unified "
        "`gfields_tomography` / `gfields_prb` integrators (continuous LOD, "
        "steerable sub-bands, Gabor kernels), the `gfields.train` and "
        "`gfields.render` pipeline scripts, and the `gfields` auxiliary CLI "
        "(viewer, voxelizer, decomposition, analysis, QA checks)."
    ),
    url="https://github.com/Arcanous98/GaborVolumes",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(
        exclude=[
            "tests*",
            "examples*",
            "scripts*",
            "paper_figures*",
            "docs*",
            "submodules*",
            "data*",
            "results*",
        ],
    ),
    entry_points={
        "console_scripts": [
            "gfields = gfields.cli:main",
            "gfields-render = gfields.render:main",
        ],
    },
    install_requires=[
        "numpy>=1.22",
        "matplotlib>=3.5",
        "pandas>=1.4",
        "plyfile>=0.7",
        "scipy>=1.9",
        "tqdm>=4.60",
        "Pillow>=9.0",
        "ipywidgets>=8.0",
        "rich>=12.0",
        "setuptools>=61",
        "colorist>=0.4",     # coloured terminal output in the training loop
    ],
    extras_require={
        "test": ["pytest>=7.0"],
        "metrics": ["scikit-image>=0.19", "torch>=1.13", "lpips>=0.1.4", "flip-evaluator>=1.0"],
        "training": ["torch>=1.13", "wandb>=0.15"],
        "viewer": ["polyscope>=2.2"],  # interactive Polyscope viewer (gfields viewer)
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
    ],
)
