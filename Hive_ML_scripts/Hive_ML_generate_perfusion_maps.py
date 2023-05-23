#!/usr/bin/env python

import importlib.resources
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent

from Hive.utils.file_utils import subfolders
from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,
    DEBUG
)
from tqdm import tqdm

import Hive_ML.configs
from Hive_ML.feature_generation.perfusion_features import PERFUSION_FUNCTIONS

DESC = dedent(
    """
    Script to generate Perfusion Maps for a given dataset. The Perfusion Maps to create, and their correpsonding suffix files, are specified in the
    ``config-file``
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -i /path/to/data_folder --config-file config_file.json
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
        help="Dataset folder",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration file path, containing training and processing parameters.",
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

    try:
        with open(arguments["config_file"]) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive_ML.configs, arguments["config_file"]) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    perfusion_maps_dict = config_dict["perfusion_maps"]
    labels = subfolders(arguments["data_folder"], join=False)

    n_workers = "1"
    if arguments["n_workers"] is None:
        if "N_THREADS" in os.environ is not None:
            n_workers = str(os.environ["N_THREADS"])
    else:
        n_workers = str(arguments["n_workers"])

    pool = Pool(int(n_workers))
    perfusion_maps = []
    for label in labels:
        subjects = subfolders(Path(arguments["data_folder"]).joinpath(label), join=False)
        for subject in subjects:
            logger.log(DEBUG, "Processing {}".format(subject))
            for perfusion_map in perfusion_maps_dict:
                if type(perfusion_maps_dict[perfusion_map]) == dict:
                    map_suffix = perfusion_maps_dict[perfusion_map]["suffix"]
                    kwargs = perfusion_maps_dict[perfusion_map]["kwargs"]
                else:
                    map_suffix = perfusion_maps_dict[perfusion_map]
                    kwargs = {}

                perfusion_maps.append(pool.starmap_async(PERFUSION_FUNCTIONS[perfusion_map],

                                                         (
                                                             (
                                                                 str(Path(arguments["data_folder"]).joinpath(label,
                                                                                                             subject,
                                                                                                             subject +
                                                                                                             config_dict[
                                                                                                                 "image_suffix"])),
                                                                 str(Path(arguments["data_folder"]).joinpath(
                                                                     label, subject,
                                                                     subject + config_dict["mask_suffix"])),
                                                                 str(Path(
                                                                     arguments["data_folder"]).joinpath(
                                                                     label, subject,
                                                                     subject + map_suffix)),
                                                                 *kwargs,

                                                             ),),
                                                         )
                                      )

    for res in tqdm(perfusion_maps, desc="Perfusion Maps Creation"):
        _ = res.get()


if __name__ == "__main__":
    main()
