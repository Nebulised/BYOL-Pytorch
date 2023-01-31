import mlflow
import yaml


def get_params(path : str):
    with open(path,
              "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as yaml_error:
            print(yaml_error)


def log_param_dicts(param_dict, existing_key = None):

    for key, val in param_dict.items():
        current_concat_key = f"{existing_key}_{key}" if existing_key is not None else key
        if type(val) is dict:
            log_param_dicts(val, existing_key = current_concat_key)
        else:
            if "every" not in current_concat_key:
                mlflow.log_param(current_concat_key, val)
