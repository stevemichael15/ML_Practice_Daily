import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
df["target"] = diabetes.target
x = df["bmi"] # independent variable
y = df.target # target variable(dependent)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state= 1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
model.fit(x_train, y_train)
coefficient = model.coef_ #Coefficient (slope, representing how much 𝑌 changes per unit change in 𝑋)
intercept = model.intercept_
y_pred = model.predict(x_test)


# to visualize the predicted data
plt.scatter(x_test, y_test, color= "black", label= "Actual")
plt.plot(x_test, y_pred, color= "blue", linewidth= 3, label= "Predicted linear regression")
plt.xlabel("BMI")
plt.ylabel("One year progression target")
plt.title("Linear regression on diabetes data")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(root_mean_squared_error(y_test, y_pred))
