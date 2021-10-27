#!/usr/bin/env python

import datetime
import importlib.resources
import json
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent

import pandas as pd
import radiomics
from radiomics import featureextractor
from tqdm import tqdm

import Radiomics_4D.configs
from Radiomics_4D.extraction.feature_extraction import extract_features_for_image_and_mask
from Radiomics_4D.utilities.file_utils import subfolders
from Radiomics_4D.utilities.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, INFO

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Run 4D Radiomics Feature extraction for the specified dataset. The Dataset tree should be in the following format:
    ::
        -Dataset_folder
        --Class_0
        ---Subject_ID_0
        ----Subject_ID_0_image.nii.gz
        ----Subject_ID_0_mask.nii.gz
        ---Subject_ID_1
        ----Subject_ID_1_image.nii.gz
        ----Subject_ID_1_mask.nii.gz
        --Class_1
        ---Subject_ID_2
        ----Subject_ID_2_image.nii.gz
        ----Subject_ID_2_mask.nii.gz
        ---Subject_ID_3
        ----Subject_ID_3_image.nii.gz
        ----Subject_ID_3_mask.nii.gz
    The extracted features are stored in a Pandas DataFrame and they can be saved in an Excel, CSV or Pickle format.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -i /path/to/input_data_folder --config-file Radiomics_4D_config.json  --feature-param-file params.yaml  --output-file Radiomics_features.xlsx
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "-i",
        "--data-folder",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        default="Radiomics_4D_config.json",
        help="Configuration JSON file with experiment and dataset parameters (Default: Radiomics_4D_config.json)",
    )

    pars.add_argument(
        "--feature-param-file",
        type=str,
        required=True,
        default="params.yaml",
        help="YAML Feature Paramaters filename, used to create the Radiomics Feature Extractor (Default: params.yaml).",
    )

    pars.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output filename for the extracted Feature Table. It can be an Excel (*.xlsx), CSV or Pickle (*.pkl) file format.",
    )

    pars.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=3,
        help="Number of parallel processes used when running the Feature Extraction process. (Default: 3)",
    )
    add_verbosity_options_to_argparser(pars)

    return pars


def main():
    parser = get_arg_parser()

    arguments = vars(parser.parse_args())

    logger = get_logger(
        name=Path(__file__).name,
        level=log_lvl_from_verbosity_args(arguments),
    )

    radiomics.logger.setLevel(logging.INFO)
    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Radiomics_4D.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    image_suffix = config_dict["image_suffix"]
    mask_suffix = config_dict["mask_suffix"]

    extractor = featureextractor.RadiomicsFeatureExtractor(arguments["feature_param_file"])

    logger.log(INFO, "Extraction parameters:\n\t{}".format(extractor.settings))
    logger.log(INFO, "Enabled filters:\n\t{}".format(extractor.enabledImagetypes))
    logger.log(INFO, "Enabled features:\n\t{}".format(extractor.enabledFeatures))

    feature_sequence_list = []
    labels = subfolders(arguments["data_folder"], join=False)
    num_threads = arguments["n_workers"]
    pool = Pool(num_threads)
    single_case_feature_extraction = []
    for label in labels:
        subjects = subfolders(Path(arguments["data_folder"]).joinpath(label), join=False)
        for subject in subjects:
            logger.log(INFO, "Extracting features for subject {}".format(subject))
            single_case_feature_extraction.append(
                pool.starmap_async(
                    extract_features_for_image_and_mask,
                    (
                        (
                            extractor,
                            str(Path(arguments["data_folder"]).joinpath(label, subject, subject + image_suffix)),
                            str(Path(arguments["data_folder"]).joinpath(label, subject, subject + mask_suffix)),
                            config_dict,
                        ),
                    ),
                )
            )

    for res in tqdm(single_case_feature_extraction, desc="Features Extraction"):
        subject_feature_sequence_list = res.get()
        for subject_feature_sequence in subject_feature_sequence_list:
            feature_sequence_list.append(subject_feature_sequence)

    features_df = pd.DataFrame()
    for feature_sequence in feature_sequence_list:
        features_df = features_df.append(feature_sequence, ignore_index=True)

    if arguments["output_file"].endswith(".xlsx"):
        features_df.to_excel(Path(arguments["data_folder"]).joinpath(arguments["output_file"]))
    elif arguments["output_file"].endswith(".csv"):
        features_df.to_csv(str(Path(arguments["data_folder"]).joinpath(arguments["output_file"])))
    elif arguments["output_file"].endswith(".pkl"):
        features_df.to_pickle(str(Path(arguments["data_folder"]).joinpath(arguments["output_file"])))
    else:
        raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")


if __name__ == "__main__":
    main()
