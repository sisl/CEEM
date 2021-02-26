#
# File: setup.py
#
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="CEEM",
    version="0.0.2",
    packages=find_packages(),
    description="Official implementation of CE-EM algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kunal Menda, Jayesh K. Gupta, Jean de Becdelièvre",
    author_email="",
    url="https://github.com/sisl/CEEM",
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
    python_requires=">=3.6",
    test_requires=["pytest"],
    zip_safe=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],)
