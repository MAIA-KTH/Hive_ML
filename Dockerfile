#Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# Contains pytorch, torchvision, cuda, cudnn
FROM continuumio/anaconda3


# Install some tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y \
 git \
 wget \
 build-essential

# Install own code
COPY ./requirements.txt .
RUN mkdir -p /opt/code/Hive_ML \
  && pip install -r requirements.txt  --pre

WORKDIR /opt/code/Hive_ML
COPY ./Hive_ML ./Hive_ML
COPY    ./MLproject .
COPY    ./Hive_ML_scripts ./Hive_ML_scripts
COPY    ./VERSION .
COPY    ./setup.py .
COPY    ./main.py .
COPY    ./README.md .
COPY    ./requirements.txt .
COPY ./docs/source/apidocs/configs/Hive_ML_config_template.json ./docs/source/apidocs/configs/Hive_ML_config_template.json

RUN pip install -v -e .
