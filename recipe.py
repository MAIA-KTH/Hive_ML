#!/usr/bin/env python
import hpccm

hpccm.config.set_working_directory('/opt/code/Hive_ML')

Stage0 += baseimage(image='continuumio/anaconda3:2023.03-1')
Stage0 += gnu()

Stage0 += copy(src=[
    'requirements.txt',
    'MLproject',
    'setup.cfg',
    'versioneer.py',
    '.gitattributes',
    'setup.py',
    'main.py',
    'MANIFEST.in',
    'README.md'
], dest='/opt/code/Hive_ML', _mkdir=True)
Stage0 += copy(src=[
    './docs/source/apidocs/configs/Hive_ML_config_template.json'],
    dest='/opt/code/Hive_ML/docs/source/apidocs/configs/Hive_ML_config_template.json', _mkdir=True)
Stage0 += copy(src=[
    './Hive_ML/'], dest='/opt/code/Hive_ML/Hive_ML/')
Stage0 += copy(src=[
    './scripts/'], dest='/opt/code/Hive_ML/scripts/')
Stage0 += workdir(directory="/opt/code/Hive_ML")
Stage0 += pip(ospackages=[""], packages=["/opt/code/Hive_ML"])
Stage0 += runscript(commands=[''])
