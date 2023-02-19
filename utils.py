import argparse

import torch

try:
    import mlflow
except:
    pass
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
    EXCLUDED_STRINGS = ("every", "mlflow")
    for key, val in param_dict.items():
        current_concat_key = f"{existing_key}_{key}" if existing_key is not None else key
        if type(val) is dict:
            log_param_dicts(val,
                            existing_key=current_concat_key)
        else:
            if not any(excluded_string in current_concat_key for excluded_string in EXCLUDED_STRINGS):
                mlflow.log_param(current_concat_key,
                                 val)


class TrainingTracker:
    """Utility class for storing metrics during training/validation

    Each "step" is an epoch
    Has optional mlflow integration
    Automatically displays out average for that particular epoch for all logged metrics

    Attributes:
        current_epoch :
            The current epoch, set to 0 at initialisation. Incremented by the increment_epoch method
        params:
            params dictionary containing recorded values for that epoch
            Keys are metric, value is list of logged values
        mlflow_enabled:
            bool to indicate whether metrics should be logged to mlflow

    """

    def __init__(self,
                 mlflow_enabled : bool =False):
        """ Initialiser method


        Args:
            mlflow_enabled:
                bool to indicate whether metrics should be logged to mlflow
        """
        self.params = {}
        self.current_epoch = 0
        self.mlflow_enabled = mlflow_enabled


    def log_metric(self, metric_name : str, value : float):
        """Method to record metric in epoch dictionary

        Args:
            metric_name:
                Name of metric value will be recorded under
            value:
                Value of the metric

        Returns:
            None

        """

        # Check metric already been logged once. If not initialise it in dict
        if metric_name not in self.params:
            self.params[metric_name] = []

        self.params[metric_name].append(value)

    def get_average_metrics(self):
        """Returns dict with values being the averages for the key(the metric)

        Returns:
            dict :
                dict[metric : average value recorded]
        """
        return {param :  sum(values)/len(values) for param, values in self.params.items()}

    def display_tracked_metrics(self):
        """ Prints out all recorded params in a single line human-readable foramt

        Returns:
            None
        """
        output_string = " | ".join([f"{param} : {average_value}" for param,average_value in self.get_average_metrics().items()])
        print(f"Epoch : {self.current_epoch} | {output_string}")


    def _log_metrics_to_mlflow(self):
        """Private method to log all recorded metrics to mlflow

        Returns:
            None

        """
        for param, average_value in self.get_average_metrics().items():
            mlflow.log_metric(param, average_value, step=self.current_epoch)

    def increment_epoch(self):
        """Method to be used at end of epoch. Prints out params and resets dictionairy values.

        Returns:
            None
        """
        self.current_epoch += 1
        self.display_tracked_metrics()
        if self.mlflow_enabled : self._log_metrics_to_mlflow()
        self._reset_dict_values()


    def _reset_dict_values(self):
        """Private method to reset only the values of the metrics dict (self.params)

        Returns:
            None
        """
        # Only resets values to lists not keys as well as keys are assumed to be more than likely the same across epochs
        for key in self.params.keys():
            self.params[key] = []


def update_yaml_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-file-path",
                        type=str,
                        help="Path to yaml file to change param within",
                        required=True)
    parser.add_argument("--param",
                        type=str,
                        help="Yaml file param to change. Split sub-sections via an -",
                        required=True)
    parser.add_argument("--new-val",
                        type = str,
                        required=True)

    params = parser.parse_args()
    with open(params.yaml_file_path, "r") as yaml_file:
        file_data = yaml.safe_load(yaml_file)
    keys = params.param.split("-")
    new_data = file_data
    for key in keys[:-1]:
        new_data = new_data[key]

    var_type = type(new_data[keys[-1]])
    new_data[keys[-1]] = var_type(params.new_val)

    with open(params.yaml_file_path, "w") as yaml_file:
        yaml.safe_dump(file_data, yaml_file)



class CosineAnnealingLRWithWarmup:

    def __init__(self,
                 optimiser : torch.optim.Optimizer,
                 warmup_epochs : int,
                 num_epochs_total : int,
                 last_epoch : int = -1,
                 verbose = False,
                 cosine_eta_min : float =0.0):
        self._current_scheduler = None
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimiser,
                                                                  start_factor=1/warmup_epochs,
                                                                  end_factor=1.0,
                                                                  total_iters=warmup_epochs,
                                                                  last_epoch=last_epoch,
                                                                  verbose=verbose)
        self.cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser,
                                                                                     T_max=num_epochs_total - warmup_epochs,
                                                                                     eta_min=cosine_eta_min,
                                                                                     last_epoch = last_epoch)
        self.epoch = 0 if last_epoch < 0 else last_epoch
        self._update_current_scheduler()


    def _update_current_scheduler(self):
        self._current_scheduler =  self.warmup_scheduler if self.epoch < self.warmup_epochs else self.cosine_annealing_scheduler

    def get_last_lr(self):
        return self._current_scheduler.get_last_lr()

    def print_lr(self, is_verbose : bool, group , lr : float, epoch=None):
        self._current_scheduler.print_lr(is_verbose=is_verbose,
                                         group=group,
                                         lr=lr,
                                         epoch=epoch)

    def step(self):
        self._update_current_scheduler()
        self._current_scheduler.step()
        self.epoch += 1

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self):
        raise NotImplementedError()
