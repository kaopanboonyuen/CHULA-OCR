"""
CHULA-OCR Setup

Author: Teerapong Panboonyuen (Kao), Ph.D.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="chula-ocr",
    version="1.0.0",
    author="Teerapong Panboonyuen (Kao)",
    author_email="teerapong.panboonyuen@gmail.com",
    description=(
        "CHULA-OCR: Uncertainty-Aware Robust Text Recognition "
        "for National-Scale Land Title Deed Digitization"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://kaopanboonyuen.github.io/CHULA-OCR/",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "OCR", "Thai NLP", "document understanding",
        "uncertainty estimation", "land title deed",
        "deep learning", "transformer",
    ],
    project_urls={
        "Paper": "https://kaopanboonyuen.github.io/CHULA-OCR/",
        "Demo": "https://kaopanboonyuen.github.io/CHULA-OCR/",
        "Bug Reports": "https://github.com/kaopanboonyuen/CHULA-OCR/issues",
        "Source": "https://github.com/kaopanboonyuen/CHULA-OCR",
    },
)
