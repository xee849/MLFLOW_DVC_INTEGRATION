import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(path,version):
    data = pd.read_csv(os.path.join(path,f"raw_data_of_version_{version}.csv"),index_col=0)
    X = data.drop(['quality'],axis=1)
    Y = data['quality']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    return x_train,y_train,x_test,y_test
