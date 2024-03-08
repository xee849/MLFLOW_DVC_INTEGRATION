
# MLflow DVC Integration

In this project mlflow and dvc are integrated in which we know how data present on remote server and we can use different version data by using dvc.api and show in mlflow artifacts and also store in artifacts.


## Create Envoirement

First create an Envoirement either by using conda or by pip
``` bash
    conda create -n <envoirement name> python=3.10.12
```

## Clone Repository

Clone the project

```bash
  https://github.com/xee849/MLFLOW_DVC_INTEGRATION.git
```

Go to the project directory

```bash
  cd MLFLOW_DVC_INTEGRATION
```

## Install Requirements

Install requiremets by using pip3,pip or conda 

```bash
  pip install -r requirements.txt
```
```bash
  pip3 install -r requirements.txt
```
```bash
  conda install -r requirements.txt
```
    
## DVC Data stored on Remote storage

Here i take an example of wine quality which is taken from http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv.
how to make versions of this data and save on remote storage using DVC.Explain in:https://github.com/xee849/DVC_ON_VM.git .

## Change Parameters

### Parameters for paramiko library to download data from remote storage in SFTP mode
To run this project first create remote data storage by using DVC data stored on remote section and take credentials of used in this section credentials are as follow:

- <foldername>/file_name.ext present in folder of DVC repo which is used for data retrieving 
- Repo address ended with .git
- IP address of VM which is used as remote storage
- SSH port used in VM mostly 22
- Remote storage username
- remote storage password

These parameters are only used for downloading data from remote storage VM to use in project 
remote storage password
### Parameters for MLFLOW where all logs will save

To Visualize experiment result on gitlab VM require following credentials:

- Token of gitlab profile
- gitlab api link which is present in repo/doc.params.yaml

before running this project change project/doc/params.yaml
all values change accoring to the mention general links
## Change Paths

In main.py change the paths for 
- parameter file path in your local machine
- save raw data
also show in main.py change accoring to your machine path
## RUN Project

After adding credentials in your params.yaml file and change path in main.py then run the comand
Before running the project set your experiment name in main.py file 
after changing all variables run command like this shown below:

```bash
    python main.py <alpha_value> <l1_ratio> <user_run_name>
```
In project we use Elastic net so for this two parameters are used for Elastic net
- alpha_value between 0 to 1 (if not given take 0.5)
- l1_ratio between 0 to 1 (if not given take 0.5)
Third argument in above command is your custom run name otherwise it take experiment name

## Results of MLFLOW 

To check MLFLOW logging go to the gitlab link which is in this for 

- For gitlab account on gitlab.com
        https://gitlab.com/<username/<reponame>/-/ml/experiments

- for gitlab server

    http://<your_endpoint>/<group/username>/<projectnameingitlabserver>/-/ml/experiments
# NOTE

As this project is used for gitlab or gitlab server to use Virtual machine as server follow following steps to get experiment data on VM

- create a 5000 port in VM firewall in inbound section with name of mlflow
- Install python in VM
- Install Mlflow in VM
- install mlflow[ssh] in your local machine
- RUN command
```bash
    mlflow server --host 0.0.0.0 --port 5000
```
- Then change credentials in params.yaml file for uri http://<IP_address_of_vm>:5000

- In main.py give artifact path like "ssh://<IP_address_of_vm>:5000/<path>"
this path in ur VM

- But for gitlab artifact path remain empty like this ''