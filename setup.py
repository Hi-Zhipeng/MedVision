"""
Setup script for MedVision.
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define core requirements
requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "pytorch-lightning>=2.0.0",
    "lightning>=2.0.0",
    "numpy>=1.21.0",
    "pyyaml>=6.0",
    "tqdm>=4.64.0",
    "click>=8.0.0",
    "pillow>=9.0.0",
    "scikit-image>=0.19.0",
    "pandas>=1.4.0",
]

# Optional dependencies
extras_require = {
    "medical": [
        "nibabel>=3.2.0",
        "SimpleITK>=2.2.0",
        "pydicom>=2.3.0",
        "vtk>=9.2.0",
    ],
    "transforms": [
        "monai>=1.3.0",
        "torchio>=0.18.0",
        "albumentations>=1.3.0",
        "opencv-python>=4.7.0",
    ],
    "visualization": [
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.10.0",
        "tensorboard>=2.10.0",
        "wandb>=0.13.0",
    ],
    "metrics": [
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "surface-distance>=0.1.0",
        "hausdorff>=0.2.6",
    ],
    "export": [
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
        "pre-commit>=2.20.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=0.18.0",
    ],
    "all": [
        # Medical
        "nibabel>=3.2.0",
        "SimpleITK>=2.2.0",
        "pydicom>=2.3.0",
        "vtk>=9.2.0",
        # Transforms
        "monai>=1.3.0",
        "torchio>=0.18.0",
        "albumentations>=1.3.0",
        "opencv-python>=4.7.0",
        # Visualization
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.10.0",
        "tensorboard>=2.10.0",
        "wandb>=0.13.0",
        # Metrics
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "surface-distance>=0.1.0",
        "hausdorff>=0.2.6",
        # Models
        "segmentation-models-pytorch>=0.5.0",
        # Export
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
    ]
}

setup(
    name="medvision",
    version="0.1.1",
    author="weizhipeng",
    author_email="weizhipeng@shu.edu.cn",
    description="A medical image segmentation framework based on PyTorch Lightning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hi-Zhipeng/MedVision",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/medvision/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "MedVision=medvision.cli.main:main",
        ],
    },
)
