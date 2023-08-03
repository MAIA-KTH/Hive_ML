FROM continuumio/anaconda3:2023.03-1

# GNU compiler
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        gfortran && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt \
    MLproject \
    setup.cfg \
    versioneer.py \
    .gitattributes \
    setup.py \
    main.py \
    MANIFEST.in \
    README.md \
    /opt/code/Hive_ML/

COPY ./docs/source/apidocs/configs/Hive_ML_config_template.json \
    /opt/code/Hive_ML/docs/source/apidocs/configs/Hive_ML_config_template.json/

COPY ./Hive_ML/* \
    /opt/code/Hive_ML/Hive_ML/

COPY ./Hive_ML_scripts/* \
    /opt/code/Hive_ML/Hive_ML_scripts/

WORKDIR /opt/code/Hive_ML

# pip
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         && \
    rm -rf /var/lib/apt/lists/*
RUN pip --no-cache-dir install /opt/code/Hive_ML

ENTRYPOINT []


