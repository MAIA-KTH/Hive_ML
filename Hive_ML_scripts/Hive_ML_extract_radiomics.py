#!/usr/bin/env python


import datetime
import json
import logging
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from importlib.resources import as_file, files
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent

import pandas as pd
import radiomics
from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import get_logger, add_verbosity_options_to_argparser, log_lvl_from_verbosity_args, INFO
from radiomics import featureextractor
from tqdm import tqdm

import Hive_ML.configs
from Hive_ML.extraction.feature_extraction import extract_features_for_image_and_mask

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to extract Radiomics for a specified dataset. The images and masks used to extract the features are specified in the
    ``config-file``.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -i /path/to/data_folder --config-file config_file.json --feature-param-file radiomic_config_file.yaml --output-file features.csv
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
        help="Configuration file path, containing training and processing parameters.",
    )

    pars.add_argument(
        "--feature-param-file",
        type=str,
        required=True,
        help="File containing configuration parameters to set up the Radiomics extractor.",
    )

    pars.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path where to store the extracted Radiomics. Available extensions: ``.xlsx``, ``.csv`` and ``.pkl``",
    )

    pars.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=None,
        help="Number of parallel threads to use when generating the Perfusion Maps (Default: ``N_THREADS``).",
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

    if Path(arguments["output_file"]).is_file():
        output_file = arguments["output_file"]
        print(f"{output_file} already present. Skipping Feature Extraction")

        return

    radiomics.logger.setLevel(logging.INFO)
    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with as_file(files(Hive_ML.configs).joinpath(arguments["config_file"])) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(arguments["feature_param_file"])
    except:
        with as_file(files(Hive_ML.configs).joinpath(arguments["feature_param_file"])) as file:
            extractor = featureextractor.RadiomicsFeatureExtractor(str(file))

    image_suffix = config_dict["image_suffix"]
    mask_suffix = config_dict["mask_suffix"]

    logger.log(INFO, 'Extraction parameters:\n\t{}'.format(extractor.settings))
    logger.log(INFO, 'Enabled filters:\n\t{}'.format(extractor.enabledImagetypes))
    logger.log(INFO, 'Enabled features:\n\t{}'.format(extractor.enabledFeatures))

    feature_sequence_list = []
    labels = subfolders(arguments["data_folder"], join=False)
    n_workers = "1"
    if arguments["n_workers"] is None:
        if "N_THREADS" in os.environ is not None:
            n_workers = str(os.environ["N_THREADS"])
    else:
        n_workers = str(arguments["n_workers"])
    pool = Pool(int(n_workers))
    single_case_feature_extraction = []

    for label in labels:
        subjects = subfolders(Path(arguments["data_folder"]).joinpath(label), join=False)
        for subject in subjects:
            distance_map_filename = None
            if "include_depth" in config_dict and config_dict["include_depth"]:
                distance_map_filename = str(Path(arguments["data_folder"]).joinpath(label, subject, subject +
                                                                                    config_dict["perfusion_maps"][
                                                                                        "distance_map"]))
            if type(image_suffix) is list:
                image_list = [
                    str(Path(arguments["data_folder"]).joinpath(label, subject, subject + single_image_suffix)) for
                    single_image_suffix in image_suffix]
                single_case_feature_extraction.append(pool.starmap_async(extract_features_for_image_and_mask,
                                                                         (
                                                                             (
                                                                                 extractor,
                                                                                 image_list,
                                                                                 str(Path(
                                                                                     arguments["data_folder"]).joinpath(
                                                                                     label, subject,
                                                                                     subject + mask_suffix)),
                                                                                 config_dict,
                                                                                 distance_map_filename
                                                                             ),
                                                                         ),
                                                                         )
                                                      )
            else:
                single_case_feature_extraction.append(pool.starmap_async(extract_features_for_image_and_mask,
                                                                         (
                                                                             (
                                                                                 extractor,
                                                                                 str(Path(
                                                                                     arguments["data_folder"]).joinpath(
                                                                                     label, subject,
                                                                                     subject + image_suffix)),
                                                                                 str(Path(
                                                                                     arguments["data_folder"]).joinpath(
                                                                                     label, subject,
                                                                                     subject + mask_suffix)),
                                                                                 config_dict,
                                                                                 distance_map_filename
                                                                             ),
                                                                         ),
                                                                         )
                                                      )

    for res in tqdm(single_case_feature_extraction, desc="Features Extraction"):
        subject_feature_sequence_list = res.get()
        for subject_feature_sequence in subject_feature_sequence_list:
            feature_sequence_list.append(subject_feature_sequence)

    features_df = []
    for feature_sequence in feature_sequence_list:
        for sequence in feature_sequence:
            features_df.append(sequence)
    features_df = pd.DataFrame.from_records(features_df)

    features_df.fillna(0, inplace=True)
    if arguments["output_file"].endswith(".xlsx"):
        features_df.to_excel(Path(arguments["output_file"]))
    elif arguments["output_file"].endswith(".csv"):
        features_df.to_csv(str(Path(arguments["output_file"])))
    elif arguments["output_file"].endswith(".pkl"):
        features_df.to_pickle(str(Path(arguments["output_file"])))
    else:
        raise ValueError("Output file format not recognized, expected one of: '.xslx', '.csv', '.pkl' ")


if __name__ == "__main__":
    main()
