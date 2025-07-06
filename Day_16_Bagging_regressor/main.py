import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_regression
x, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
lr_reg = LinearRegression()
dt_reg = DecisionTreeRegressor()
sv_reg = SVR(kernel= "linear")
from sklearn.ensemble import VotingRegressor
model = VotingRegressor(estimators=[("Linear_Regression", lr_reg), ("Decision_Tree", dt_reg), ("Support_Vector", sv_reg)], verbose=2)
model.fit(x_train, y_train)
from sklearn.metrics import r2_score
print(r2_score(y_test, model.predict(x_test)))