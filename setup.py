#
# File: setup.py
#
from setuptools import setup, find_packages

setup(
    name="CEEM",
    version="0.0.1",
    packages=find_packages(),
    description="",
    install_requires=[
        "click",
        "joblib",
        "pandas",
        "matplotlib",
        "numpy",
        "scipy",
        "torch",
        "termcolor",
        "python_dateutil",
        "tensorboard",
        "future",
        "tqdm",
        "pytest",
    ],
    test_requires=["pytest"],
    zip_safe=True,)
