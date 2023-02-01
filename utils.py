import mlflow
import yaml


def get_params(path: str):
    """ Loads in yaml file and returns dict of contents
    Args:
        path:
            Full path to yaml file

    Returns:
        dict : Contents of yaml file

    """
    with open(path,
              "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def log_param_dicts(param_dict : dict,
                    existing_key=None):
    """Method for logging mlflow params for dict of dicts

    Recursively logs params for the dicts. Stop case is when no longer anymore nested dicts to recurse through
    Automatically adds the key as a prefix via the param "existing_key"

    Args:
        param_dict:
            An N-level nested dictionairy
        existing_key:
            The prefix added onto the key for the params

    Returns:
        None
    """
    for key, val in param_dict.items():
        current_concat_key = f"{existing_key}_{key}" if existing_key is not None else key
        if type(val) is dict:
            log_param_dicts(val,
                            existing_key=current_concat_key)
        else:
            if "every" not in current_concat_key:
                mlflow.log_param(current_concat_key,
                                 val)
