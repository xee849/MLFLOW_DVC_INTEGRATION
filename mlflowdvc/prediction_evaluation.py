from mlflowdvc.utils.common import eval_metrics

def model_prediction(model,x_test):
    y_predict = model.predict(x_test)
    return y_predict
def evaluation(actual,predict):
    rmse, mae, r2 = eval_metrics(actual=actual, pred=predict)
    return rmse, mae, r2
