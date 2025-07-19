from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dynamic-ssm-transformer",
    version="0.1.0",
    author="Julian Adam",
    author_email="jul.p.adam@gmail.com",
    description="Adaptive hybrid attention-SSM architecture with dynamic gating (Using top layer 26-30 of Gemma 3n2b base)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jada42/Dynamic-Gated-SSM-Transformer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
        "training": ["datasets", "wandb"],
        "memory": ["faiss-cpu"],
    },
)
