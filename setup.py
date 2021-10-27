import glob

import setuptools
from setuptools import setup

setup(
    name="Radiomics_4D",
    version="1.0",
    url="",
    license="",
    author="Astaraki Mehdi, Bendazzoli Simone",
    author_email="simben@kth.se",
    description="Python package to run 4D Radiomics experiments, including Feature Extraction, Feature Selection and Feature Analysis.",  # noqa: E501
    packages=setuptools.find_packages("src"),
    package_data={
        "": ["configs/*.yml", "configs/*.json"],
    },
    package_dir={"": "src"},
    install_requires=[
        "SimpleITK",
        "pyradiomics",
        "six",
        "tqdm",
        "pandas",
        "coloredlogs",
    ],
    scripts=glob.glob("scripts/*"),
)
