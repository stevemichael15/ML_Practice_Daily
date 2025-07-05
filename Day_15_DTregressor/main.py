import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data= data.data, columns= data.feature_names)
df["Price"] = data.target
df = df.sample(frac= 0.20)
x = df.drop("Price", axis = 1)
y = df["Price"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import GridSearchCV
# params = {
#     "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
#     "splitter": ["best", "random"],
#     "max_depth": [1, 2, 3, 4, 5, 10],
#     "max_features": ["sqrt", "log2"]
# }
# model = GridSearchCV(DecisionTreeRegressor(),param_grid=params, scoring= "r2", cv= 5, verbose=2)
# model.fit(x_train, y_train)
# print(model.best_params_)
model = DecisionTreeRegressor(criterion="friedman_mse", max_depth=3, max_features= "sqrt",splitter="best")
model.fit(x_train,y_train)
from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_test, model.predict(x_test)))
print(mean_squared_error(y_test, model.predict(x_test)))
