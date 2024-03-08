from sklearn.linear_model import ElasticNet

def training(x,y,alpha,l1_ratio):
    model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    model.fit(x,y)
    return model
