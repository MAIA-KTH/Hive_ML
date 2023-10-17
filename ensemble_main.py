import click
import importlib
import json
import mlflow
import os
import subprocess
import time
from jsonschema import Draft7Validator
from pathlib import Path

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
@click.option("--config-file", type=str)
@click.option("--ensemble-config", type=str)
@click.option("--experiment-name", type=str)
def workflow(config_file, experiment_name, ensemble_config):
    print("Starting Experiment")
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
            "Hive_ML_ensemble_models",
            {"feature_file": Path(os.environ["ROOT_FOLDER"]).joinpath(f"{experiment_name}.xlsx"),
             "experiment_name": experiment_name,
             "config_file": config_file,
             "ensemble_config": ensemble_config},
        )
        print(f"Hive_ML_ensemble_models {time.time() - start_time} s")


if __name__ == "__main__":
    workflow()
