from setuptools import setup, find_packages

setup(
    name="descartes-pharma",
    version="1.2.0",
    description="Mechanistic Zombie Detection for Drug Discovery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "chem": ["rdkit>=2023.03.1", "PyTDC>=0.4.0"],
        "neuro": ["allensdk>=2.15.0"],
        "alphafold": ["biopython>=1.81"],
        "llm": ["anthropic>=0.18.0"],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "all": [
            "rdkit>=2023.03.1", "PyTDC>=0.4.0",
            "matplotlib>=3.7.0", "seaborn>=0.12.0",
            "anthropic>=0.18.0",
        ],
    },
)
