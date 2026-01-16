from setuptools import setup, find_packages

setup(
    name="HYDRA",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "pandas",
        "tqdm",
        "matplotlib",
    ],
)
