import importlib
import json
import os
import subprocess
import time
from pathlib import Path

import click
import mlflow
from jsonschema import Draft7Validator

import Hive_ML.configs


def check_json_config(config_dict):
    with open("docs/source/apidocs/configs/Hive_ML_config_template.json", "r") as f:
        schema = json.load(f)

    validator = Draft7Validator(schema)

    errors = sorted(validator.iter_errors(config_dict), key=lambda e: e.path)

    if len(errors) == 0:
        return True
    else:
        print("Found errors while validating JSON Configuration file: ")
        for error in errors:
            print(error.message)
        return False


def inject_known_hosts():
    try:
        ssh_path = os.environ["SSH_PATH"]
        Path(ssh_path).mkdir(parents=True, exist_ok=True)
        host_key = os.environ["MLFLOW_BUCKET_HOSTKEY"]
        bucket_hostname = os.environ["MLFLOW_BUCKET_HOSTNAME"]
        bucket_port = os.environ["MLFLOW_BUCKET_PORT"]
        user = os.environ["MLFLOW_BUCKET_USERNAME"]
        ssh_key_path = os.environ["MLFLOW_SSH_KEY_PATH"]

        with open(str(Path(ssh_path).joinpath("known_hosts")), "w") as f:
            subprocess.run(["echo", f"{host_key}"], stdout=f)

        with open(str(Path(ssh_path).joinpath("config")), "w") as f:
            subprocess.run(["echo", f"Host {bucket_hostname}"], stdout=f)
            subprocess.run(["echo", f"  HostName k8s-maia.com"], stdout=f)
            subprocess.run(["echo", f"  Port {bucket_port}"], stdout=f)
            subprocess.run(["echo", f"  User {user}"], stdout=f)
            subprocess.run(["echo", f"  IdentityFile {ssh_key_path}"], stdout=f)
            return True
    except:
        return False


def _get_or_run(entrypoint, parameters):
    print("Launching new run for entrypoint={} and parameters={}".format(entrypoint, parameters))

    cl = [entrypoint]
    for parameter in parameters:
        cl.append("--{}".format(parameter.replace("_", "-")))
        if type(parameters[parameter]) != list:
            cl.append("{}".format(parameters[parameter]))
        else:
            _ = [cl.append("{}".format(param)) for param in parameters[parameter]]
    subprocess.call(cl)


@click.command()
@click.option("--data-folder", type=str)
@click.option("--config-file", type=str)
@click.option("--experiment-name", type=str)
@click.option("--radiomic-config-file", type=str)
def workflow(data_folder, config_file, experiment_name, radiomic_config_file):
    print("Starting Experiment")
    log_artifacts = inject_known_hosts()
    try:
        with open(config_file) as json_file:
            config_dict = json.load(json_file)
    except FileNotFoundError:
        with importlib.resources.path(Hive_ML.configs, config_file) as json_path:
            with open(json_path) as json_file:
                config_dict = json.load(json_file)

    if not check_json_config(config_dict):
        return
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]) as run:

        start_time = time.time()
        _get_or_run(
            "Hive_ML_extract_radiomics", {"data_folder": data_folder, "feature_param_file": radiomic_config_file,
                                          "config_file": config_file,
                                          "output_file": Path(os.environ["ROOT_FOLDER"]).joinpath(
                                              f"{experiment_name}.xlsx")},
        )

        if log_artifacts:
            mlflow.log_artifact(str(Path(os.environ["ROOT_FOLDER"]).joinpath(f"{experiment_name}.xlsx")))
        print(f"Hive_ML_extract_radiomics took {time.time() - start_time} s")

        start_time = time.time()
        _get_or_run(
            "Hive_ML_feature_selection",
            {"feature_file": Path(os.environ["ROOT_FOLDER"]).joinpath(f"{experiment_name}.xlsx"),
             "experiment_name": experiment_name,
             "config_file": config_file},
        )

        print(f"Hive_ML_feature_selection took {time.time() - start_time} s")
        fs_summary_file_path = Path(os.environ["ROOT_FOLDER"]).joinpath(
            experiment_name, config_dict["feature_selection"], config_dict["feature_aggregator"],
            "FS", f"{experiment_name}_FS_summary.json")
        if log_artifacts:
            mlflow.log_artifact(str(fs_summary_file_path))

        start_time = time.time()
        _get_or_run(
            "Hive_ML_model_fitting",
            {"feature_file": Path(os.environ["ROOT_FOLDER"]).joinpath(f"{experiment_name}.xlsx"),
             "experiment_name": experiment_name,
             "config_file": config_file},
        )
        print(f"Hive_ML_model_fitting took {time.time() - start_time} s")

        if log_artifacts:
            model_fitting_folder = str(Path(os.environ["ROOT_FOLDER"]).joinpath(
                experiment_name))

            mlflow.log_artifact(model_fitting_folder)


if __name__ == "__main__":
    workflow()
