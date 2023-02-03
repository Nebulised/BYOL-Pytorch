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
        for param, average_value in self.get_average_metrics():
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
