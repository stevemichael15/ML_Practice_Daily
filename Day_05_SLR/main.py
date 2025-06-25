import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("height-weight.csv")
# problem statement >> To predict height based on weight
x = df.Weight
y = df.Height
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
x_train = x_train.values.reshape(-1, 1)  # Convert to 2D array
x_test = x_test.values.reshape(-1, 1)  # Convert to 2D array
# to remember
# scaling should be done always after train_test_split
# target variable should not be scaled
# avoid scaling categorical data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
coefficient = model.coef_
intercept = model.intercept_
predictor = model.predict(x_train)
# plt.scatter(x_train, y_train)
# plt.plot(x_train, predictor)
# plt.show()
y_pred_test = model.predict(x_test)
# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred_test, color= 'r')
# plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error , root_mean_squared_error, r2_score
# print(f"MSE : {mean_squared_error(y_test, y_pred_test)}")
# print(f"RMSE : {root_mean_squared_error(y_test, y_pred_test)}")
# print(f"MAE : {mean_absolute_error(y_test, y_pred_test)}")
# print(f"R2 : {r2_score(y_test, y_pred_test)}")
error = y_test - y_pred_test
print(error)
sns.distplot(error)
plt.show()
