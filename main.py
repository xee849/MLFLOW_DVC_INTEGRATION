import sys
import os
from mlflowdvc.data_prepration import DVCreadFile
from mlflowdvc.utils.common import read_yaml
from mlflowdvc import logger
from mlflowdvc.feature_extraction import split_data
from mlflowdvc.training import training
from mlflowdvc.prediction_evaluation import model_prediction,evaluation
import mlflow
from mlflow.sklearn import log_model
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models import infer_signature



"""user give his name then experiment created with his name and all runs in it done by this user"""
experiment_name = "zeeshan"



params_file_path = '<path where your project present>/MLFLOW_DVC_INTEGRATION_1/docs/params.yaml'
raw_data_storage_in_local = '<path where your project present>/MLFLOW_DVC_INTEGRATION_1/data/raw/'
config = read_yaml(params_file_path)
repo_url = config.dvc_repo_url
Data_path_in_repo = config.folder_path_in_dvc

data_version = "v0.2"
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 1 else 0.5
runname = str(sys.argv[3]) if len(sys.argv) != 0 else experiment_name




STAGE_NAME = f"DOWNLOAD RAW DATA OF VERSION {data_version}"

try:
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} STARTED <<<<<<<<<<<")
    data = DVCreadFile(path=Data_path_in_repo,repo=repo_url,version=data_version,yaml_path=params_file_path)
    source_url = data.source_url()
    data.save_raw_file(raw_data_storage_in_local)
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} COMPLETED <<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "SPLIT FEATURES AND TARGET FROM RAW DATA"

try:
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} STARTED <<<<<<<<<<<")
    X_train,Y_train,X_test,Y_test = split_data(raw_data_storage_in_local,data_version)
    print(f"SIZE OF X TRAIN IS {X_train.shape} SIZE OF X TEST IS {X_test.shape}")
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} COMPLETED <<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "TRAIN DATA ON ELASTIC NET "

try:
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} STARTED <<<<<<<<<<<")
    Model = training(X_train,Y_train,alpha,l1_ratio)
    print(Model)
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} COMPLETED <<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "PREDICTION AND EVALUATE MODEL ON RMSE,MAE AND R2  "

try:
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} STARTED <<<<<<<<<<<")
    Y_predict = model_prediction(Model,X_test)
    RMSE,MAE,R2 = evaluation(Y_test,Y_predict)
    print(f"ROOT MEAN SQUARE ERROR IS {RMSE}")
    print(f"MEAN ABSOLUTE ERROR IS {MAE}")
    print(f"R squared IS {R2}")
    logger.info(f">>>>>>>>>STAGE {STAGE_NAME} COMPLETED <<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

print("LOG PARAMETERS AND MATRICS ON MLFLOW")
"""
THERE ARE TWO WAYS TO LOG PARAMETERS ON MLFLOW SERVER IT DEPENDS WHICH SERVER U ARE USING 
1 : FOR GITLAB
    WE USE API URI AND ACCESS TOKEN 
    LIKE 
    os.environ["MLFLOW_TRACKING_TOKEN"]="<your_access_token>"
    os.environ["MLFLOW_TRACKING_URI"]="<your gitlab endpoint>/api/v4/projects/<your project id>/ml/mlflow"
2: IF MLFLOW SERVER IS ON VIRTUAL MACHINE THEN WE GIVE HTTP REQUEST WITH PORT 
i) start server on VM
ii) and here u give http://<vm ip address>:5000
mlflow use 5000 port so we give access 5000 port in VM setting in inbound traffic
"""
# here i use gitlab server for this i use gitlab credentials
os.environ["MLFLOW_TRACKING_TOKEN"]=str(config.token)
os.environ["MLFLOW_TRACKING_URI"]=str(config.uri)
mlflow.set_experiment(experiment_name=experiment_name)

with mlflow.start_run(run_name=runname):
    mlflow.log_param("DATA URI" , source_url)
    mlflow.log_param("DATA VERSION USED IN TRAINING" , data_version)
    mlflow.log_param("NUMBER OF FEATURE USED IN TRAINING" , X_train.shape[1])
    mlflow.log_param("ELASTIC NET ALPHA VALUE", alpha)
    mlflow.log_param("ELASTIC NET L1 RATIO" , l1_ratio)
    mlflow.log_metric("ROOT MEAN SQUARE ERROR" , RMSE)
    mlflow.log_metric("MEAN ABSOLUTE ERROR",MAE)
    mlflow.log_metric("R-SQUARED",R2)
    signature =infer_signature(X_train,Y_predict)
    log_model(Model,artifact_path='',signature=signature)
"""
to avoid from one by one log values u can also use mlflow.log_params and mlflow.log_metrics
for this purpose u can make dictionary of parameters and metrics 

parameters = {
    "DATA URI": source_url, "DATA VERSION USED IN TRAINING": data_version,
    "NUMBER OF FEATURE USED IN TRAINING": X_train.shape[1],
    "ELASTIC NET ALPHA VALUE": alpha, "ELASTIC NET L1 RATIO": l1_ratio
}
metrics = {"ROOT MEAN SQUARE ERROR": RMSE, "MEAN ABSOLUTE ERROR": MAE, "R-SQUARED": R2}
with mlflow.start_run(run_name=runname):
    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)
    signature = infer_signature(X_train, Y_predict)
    log_model(Model, artifact_path='', signature=signature)
"""

