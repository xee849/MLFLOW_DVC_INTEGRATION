import paramiko as pa
import dvc.api
import pandas as pd
from mlflowdvc.utils.common import read_yaml
import os


class DVCreadFile:
    def __init__(self, path, repo, version, yaml_path):
        self.path = path
        self.repo = repo
        self.version = version
        self.yamlpath = yaml_path

    def source_url(self):
        url = dvc.api.get_url(self.path, self.repo, self.version)
        return url

    def save_raw_file(self,save_path):
        config = read_yaml(self.yamlpath)
        hostname = str(config.hostname_vm)
        port = str(config.port_vm)
        username = str(config.username_vm)
        passward = str(config.password_vm)
        ssh_client = pa.SSHClient()
        ssh_client.set_missing_host_key_policy(pa.AutoAddPolicy())
        ssh_client.connect(hostname, port, username, password=passward)
        path = self.source_url()
        index = path.find('/home')
        remote_file_path = path[index:]
        sftp = ssh_client.open_sftp()
        with sftp.open(remote_file_path) as f:
            dataframe = pd.read_csv(f, sep=';')
        sftp.close()
        ssh_client.close()
        dataframe.to_csv(os.path.join(save_path,f"raw_data_of_version_{self.version}.csv"))




