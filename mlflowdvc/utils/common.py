from box import ConfigBox
import yaml
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return ConfigBox(content)

def eval_metrics(actual, pred):
    rmse = round(np.sqrt(mean_squared_error(actual, pred)),3)
    mae = round(mean_absolute_error(actual, pred),3)
    r2 = round(r2_score(actual, pred),3)
    return rmse, mae, r2