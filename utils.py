import argparse
from typing import Optional

import torch

import custom_optimisers
from networks import BYOL

try:
    import mlflow
except:
    print("Warning :  Issue importing mlflow. If you are not using mlflow ignore this")
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


def log_param_dicts(param_dict: dict,
                    existing_key=None):
    """Method for logging mlflow params for dict of dicts

    Recursively logs params for the dicts. Stop case is when no longer anymore nested dicts to recurse through
    Automatically adds the key as a prefix via the param "existing_key"

    Args:
        param_dict:
            An N-level nested dictionary
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

    Each "step" is an _epoch
    Has optional mlflow integration
    Automatically displays out average for that particular _epoch for all logged metrics

    Attributes:
        current_epoch :
            The current _epoch, set to 0 at initialisation. Incremented by the increment_epoch method
        params:
            params dictionary containing recorded values for that _epoch
            Keys are metric, value is list of logged values
        mlflow_enabled:
            bool to indicate whether metrics should be logged to mlflow

    """

    def __init__(self,
                 mlflow_enabled: bool = False):
        """ Initialiser method


        Args:
            mlflow_enabled:
                bool to indicate whether metrics should be logged to mlflow
        """
        self.params = {}
        self.current_epoch = 0
        self.mlflow_enabled = mlflow_enabled

    def log_metric(self,
                   metric_name: str,
                   value: float):
        """Method to record metric in _epoch dictionary

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
        return {param: sum(values) / len(values) for param, values in self.params.items()}

    def display_tracked_metrics(self):
        """ Prints out all recorded params in a single line human-readable foramt

        Returns:
            None
        """
        output_string = " | ".join([f"{param} : {average_value}" for param, average_value in self.get_average_metrics().items()])
        print(f"Epoch : {self.current_epoch} | {output_string}")

    def _log_metrics_to_mlflow(self):
        """Private method to log all recorded metrics to mlflow

        Returns:
            None

        """
        for param, average_value in self.get_average_metrics().items():
            mlflow.log_metric(param,
                              average_value,
                              step=self.current_epoch)

    def increment_epoch(self):
        """Method to be used at end of _epoch. Prints out params and resets dictionairy values.

        Returns:
            None
        """
        self.display_tracked_metrics()
        if self.mlflow_enabled: self._log_metrics_to_mlflow()
        self.current_epoch += 1
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
    """ Python method for updating yaml files. Designed to be called from a shell script/command line

    Updates a single yaml parameter (--param)  within yaml file specified by --yaml-file-path to be --new-val

    Returns:
        None

    """
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
                        type=str,
                        required=True)

    params = parser.parse_args()
    with open(params.yaml_file_path,
              "r") as yaml_file:
        file_data = yaml.safe_load(yaml_file)
    keys = params.param.split("-")
    new_data = file_data
    for key in keys[:-1]:
        new_data = new_data[key]

    var_type = type(new_data[keys[-1]])
    new_data[keys[-1]] = var_type(params.new_val)

    with open(params.yaml_file_path,
              "w") as yaml_file:
        yaml.safe_dump(file_data,
                       yaml_file)


class CosineAnnealingLRWithWarmup:
    """Class for combining cosine annealing learning rate with a linear warmup

    Attributes:
         _current_scheduler :
            One of torch CosineAnnealingLR or LinearLR
            Corresponds to current scheduler that should be used
            In case current_epoch < num warmup epochs
                Set to linearLR
            Else
                set to CosineAnnealingLR
        _warmup_epochs :
            The number of epochs to warm up the learning rate linearly over
        _warmup_scheduler :
            The pytorch LinearLR scheduler to perform linear warm up using
        _epoch :
            Current epoch. Determined by last_epoch
    """

    def __init__(self,
                 optimiser: torch.optim.Optimizer,
                 warmup_epochs: int,
                 num_epochs_total: int,
                 last_epoch: int = -1,
                 verbose=False,
                 cosine_eta_min: float = 0.0):
        self._current_scheduler = None
        self._warmup_epochs = warmup_epochs
        self._warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimiser,
                                                                   start_factor=1 / warmup_epochs,
                                                                   end_factor=1.0,
                                                                   total_iters=warmup_epochs,
                                                                   last_epoch=last_epoch,
                                                                   verbose=verbose)
        self._cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser,
                                                                                      T_max=num_epochs_total - warmup_epochs,
                                                                                      eta_min=cosine_eta_min,
                                                                                      last_epoch=last_epoch)
        # Pytorch uses last_epoch -1 to determine first epoch
        # We however will use epoch 0 to do this
        self._epoch = 0 if last_epoch < 0 else last_epoch
        self._update_current_scheduler()

    def _update_current_scheduler(self):
        """ Method to update current in-use scheduler

        Returns:
            None
        """
        self._current_scheduler = self._warmup_scheduler if self._epoch < self._warmup_epochs else self._cosine_annealing_scheduler

    def get_last_lr(self):
        """ Equivalent of torch scheduler get_last_lr

        Returns:
            float : schedulers last learning rate
        """
        return self._current_scheduler.get_last_lr()

    def print_lr(self,
                 is_verbose: bool,
                 group,
                 lr: float,
                 epoch=None):
        """ Equivalent of torch scheduler print lr. (Prints current learning rate)

        Args:
            is_verbose:
            group:
            lr:
            epoch:

        Returns:
            None
        """
        self._current_scheduler.print_lr(is_verbose=is_verbose,
                                         group=group,
                                         lr=lr,
                                         epoch=epoch)

    def step(self):
        """ Updates currently assigned scheduler and steps that scheduler
            Expected to be called at end of every epoch
        Returns:
            None
        """
        self._update_current_scheduler()
        self._current_scheduler.step()
        self._epoch += 1

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self):
        raise NotImplementedError()


def setup_mlflow(run_type: str,
                 mlflow_tracking_uri: str,
                 mlflow_experiment_name: str,
                 args: argparse.Namespace,
                 model_params: dict,
                 run_params: dict,
                 model_param_file_path: str,
                 run_param_file_path: str,
                 mlflow_run_id: str = None,
                 **kwargs):
    print(f"Mlflow enabled. Tracking URI : {mlflow_tracking_uri}")
    import mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    if run_type == "train":
        nested = False
        run_name = "Self-Supervised-Pre-Training"
    elif run_type == "fine-tune":
        nested = True
        run_name = "Fine-Tuning"
    else:
        nested = True
        run_name = "Evaluation"
    nested = False if mlflow_run_id is None else nested
    if nested:
        mlflow.start_run(run_id=mlflow_run_id)
    mlflow.start_run(nested=nested,
                     run_name=run_name)
    mlflow_enabled = True
    log_param_dicts(param_dict=run_params)
    log_param_dicts(param_dict=model_params,
                    existing_key="model")
    # Vars converts namespace object to dict
    log_param_dicts(param_dict=vars(args))
    for path_to_param_file in (model_param_file_path, run_param_file_path):
        mlflow.log_artifact(local_path=path_to_param_file,
                            artifact_path="parameters")


def create_optimiser(model: BYOL,
                     optimiser_params: dict,
                     run_type : str,
                     optimiser_state_dict: Optional[dict] = None,
                     freeze_encoder = False):
    def is_not_bias_or_batch_norm(param):
        return param.ndim != 1

    optimiser_type = optimiser_params.pop("type",
                                          None)
    if run_type == "train":
        parameters = torch.nn.ModuleList([model.online_encoder,
                                         model.online_projection_head,
                                         model.online_predictor]).parameters()
        if optimiser_type in ("adam", "sgd"):
            print("Removing weight decay from bias and batch norm layers for pre-training")
            parameters = [{"params" : [param for param in parameters if not is_not_bias_or_batch_norm(param)],
                          "weight_decay" : 0.0},
                          {"params" : [param for param in parameters if is_not_bias_or_batch_norm(param)],
                          "weight_decay" : optimiser_params.pop("weight_decay")}]


    elif run_type == "fine-tune":
        if freeze_encoder:
            parameters = model.fc.parameters()
        else:
            parameters = torch.nn.ModuleList([model.online_encoder, model.fc]).parameters()
    else:
        raise Exception(f"Received unexpected run type : {run_type}")

    if optimiser_type == "adam":
        optimiser = torch.optim.Adam(params=parameters,
                                     **optimiser_params)
    elif optimiser_type == "lars":
        optimiser = custom_optimisers.Lars(parameters,
                                           weight_decay_filter=is_not_bias_or_batch_norm,
                                           lars_adaptation_filter=is_not_bias_or_batch_norm,
                                           **optimiser_params)
    elif optimiser_type == "sgd":
        optimiser = torch.optim.SGD(params=parameters,
                                    **optimiser_params)
    else:
        raise Exception(f"Unexpected optimiser type : {optimiser_type}")  #

    if optimiser_state_dict is not None:
        optimiser.load_state_dict(optimiser_state_dict)

    return optimiser
