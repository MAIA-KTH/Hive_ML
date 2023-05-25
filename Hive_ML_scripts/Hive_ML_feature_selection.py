#!/usr/bin/env python

import datetime
import importlib.resources
import json
import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from Hive.utils.log_utils import (
    get_logger,
    add_verbosity_options_to_argparser,
    log_lvl_from_verbosity_args,

)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import Hive_ML.configs
from Hive_ML.data_loader.feature_loader import load_feature_set
from Hive_ML.training.models import adab_tree, random_forest, knn, decicion_tree, lda, qda, naive, svm_kernel, \
    logistic_regression, ridge, mlp
from Hive_ML.utilities.feature_utils import data_shuffling, feature_normalization

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

MODELS = {
    "rf": random_forest,
    "adab": adab_tree,
    "lda": lda,
    "qda": qda,
    "logistic_regression": logistic_regression,
    "knn": knn,
    "naive": naive,
    "decision_tree": decicion_tree,
    "svm": svm_kernel,
    "ridge": ridge,
    "mlp": mlp
}

DESC = dedent(
    """
    Script to run Sequential 5-CV Forward Feature Selection on a Feature Set.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} -feature-file /path/to/feature_table.csv --config-file config_file.json --experiment-name Radiomics
    """.format(  # noqa: E501
        filename=Path(__file__).name
    )
)
import warnings

warnings.filterwarnings("ignore")


def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--feature-file",
        type=str,
        required=True,
        help="Input Dataset folder",
    )

    pars.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration JSON file with experiment and dataset parameters (Default: LungLobeSeg_nnUNet_3D_config.json)",
    )

    pars.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="",
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

    selected_features = {}

    models = config_dict["models"]

    feature_set, subject_ids, subject_labels, feature_names, _, _, _, _ = load_feature_set(arguments["feature_file"],
                                                                                           get_4D_stats=False,
                                                                                           flatten_features=True)

    features = feature_set

    n_features = features.shape[1]
    n_subjects = features.shape[0]

    filtered_feature_set = []
    filtered_feature_names = []

    for feature in range(n_features):
        exclude = False
        for feature_val in np.unique(features[:, feature]):
            if (np.count_nonzero(features[:, feature] == feature_val) / n_subjects) > 0.5:
                exclude = True
                print("Excluding:", feature_names[feature])
                break

        if not exclude:
            filtered_feature_set.append(list(features[:, feature]))
            filtered_feature_names.append(feature_names[feature])

    feature_set = np.vstack(filtered_feature_set).T
    feature_names = filtered_feature_names
    label_set = np.array(subject_labels)
    print("# Features: {}".format(feature_set.shape[1]))
    print("# Labels: {}".format(label_set.shape))

    train_feature_set, train_label_set = data_shuffling(feature_set, label_set, config_dict["random_seed"])

    experiment_name = arguments["experiment_name"]

    aggregation = "Flat"

    experiment_dir = Path(os.environ["ROOT_FOLDER"]).joinpath("Experiments",
                                                              experiment_name, config_dict["feature_selection"],
                                                              aggregation,
                                                              "FS")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    n_iterations = 0
    for classifier in models:
        if classifier in ["rf", "adab"]:
            n_iterations += 1
        else:
            n_iterations += config_dict["n_folds"]

    pbar = tqdm(total=n_iterations)
    for classifier in models:
        if classifier in ["rf", "adab"]:
            pbar.update(1)
            continue
        selected_features[classifier] = {}
        kf = StratifiedKFold(n_splits=config_dict["n_folds"], random_state=config_dict["random_seed"], shuffle=True)
        for fold, (train_index, _) in enumerate(kf.split(train_feature_set, train_label_set)):
            pbar.set_description(f"{classifier}, fold {fold} FS")
            fs_summary = Path(experiment_dir).joinpath(f"FS_summary_{classifier}_fold_{fold}.json")
            if fs_summary.is_file():
                with open(fs_summary, "r") as f:
                    selected_features[classifier][fold] = json.load(f)

            else:
                n_features = config_dict["n_features"]
                if n_features > train_feature_set.shape[1]:
                    n_features = train_feature_set.shape[1]
                x_train = train_feature_set[train_index]
                y_train = train_label_set[train_index]

                x_train, _, _ = feature_normalization(x_train)
                clf = MODELS[classifier](**models[classifier])
                sffs_model = SFS(clf,
                                 k_features=n_features,
                                 forward=True,
                                 floating=True,
                                 scoring='roc_auc',
                                 verbose=0,
                                 n_jobs=-1,
                                 cv=5)
                df_features_x = pd.DataFrame()
                for x_train_row in x_train:
                    df_row = {}
                    for idx, feature_name in enumerate(feature_names):
                        df_row[feature_name] = x_train_row[idx]
                    df_features_x = df_features_x.append(df_row, ignore_index=True)
                print(np.sum(np.isnan(df_features_x.values.shape)))
                sffs = sffs_model.fit(df_features_x, y_train)

                sffs_features = sffs.subsets_
                for key in sffs_features:
                    sffs_features[key]['cv_scores'] = sffs_features[key]['cv_scores'].tolist()

                selected_features[classifier][fold] = sffs_features
                with open(fs_summary, "w") as f:
                    json.dump(sffs_features, f)
            pbar.update(1)

    with open(str(Path(experiment_dir).joinpath(f"{experiment_name}_FS_summary.json")), "w") as f:
        json.dump(selected_features)


if __name__ == "__main__":
    main()
