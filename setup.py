import os
import versioneer

import setuptools
from setuptools import setup

def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


setup(
    version=versioneer.get_version(),
    packages=setuptools.find_packages(),
    package_data={
        "": ["configs/*.yml", "configs/*.json","tutorials/*"],
    },
    zip_safe=False,
    data_files=[('', ["requirements.txt"]), ],
    # package_dir={"": "src"},
    #install_requires=resolve_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt")),
    entry_points={
        "console_scripts": [
            "Hive_ML_extract_radiomics = Hive_ML_scripts.Hive_ML_extract_radiomics:main",
            "Hive_ML_feature_selection = Hive_ML_scripts.Hive_ML_feature_selection:main",
            "Hive_ML_generate_perfusion_maps = Hive_ML_scripts.Hive_ML_generate_perfusion_maps:main",
            "Hive_ML_model_fitting = Hive_ML_scripts.Hive_ML_model_fitting:main",

        ],
    },
    cmdclass=versioneer.get_cmdclass(),
    keywords=["machine learning", "image classification", "PCR", "medical image analysis", "DCE MRI", "radiomics",
              "feature selection", "radiodynamics"],
)
