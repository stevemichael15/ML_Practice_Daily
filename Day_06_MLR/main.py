import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target
x = df.drop("target", axis=1)
y = df.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
# do scaling HW
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
coefficient = model.coef_
intercept = model.intercept_
y_pred_test = model.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(y_test, y_pred_test))
print(r2_score(y_test, y_pred_test))
