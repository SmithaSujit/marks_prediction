import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def marks_prediction(hrs):
    X= pd.read_csv(r"C:\Users\user\Project\my_app\X_Train.csv")
    Y= pd.read_csv(r"C:\Users\user\Project\my_app\Y_Train.csv")

    X=X.values
    y=Y.values

    model = LinearRegression()
    model.fit(X,y)

    X_test= np.array(hrs, dtype=float)
    X_test= X_test.reshape((1, -1))

    return model.predict(X_test)[0]

