from setuptools import setup, find_packages

setup(
    name="le-masson-replication",
    version="0.3.0",
    description="Replication of Le Masson et al. 2002 — thalamic circuit replacement ladder",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "rung3": [
            "h5py",
            "torch",
            "torchdiffeq",
            "scikit-learn",
        ],
    },
)
