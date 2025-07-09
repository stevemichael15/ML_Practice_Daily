import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
df = sns.load_dataset("tips")
x = df.drop("total_bill", axis = 1)
y = df["total_bill"]
cat_cols = [col for col in df.columns if df[col].dtypes == "category"]
num_cols = [col for col in df.columns if df[col].dtypes != "category"]
num_cols.remove("total_bill")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
num_pipeline = Pipeline([("SimpleImputer", SimpleImputer(strategy="mean")), ("StandardScaler", StandardScaler())])
cat_pipeline = Pipeline([("SimpleImputer", SimpleImputer(strategy="most_frequent")), ("OneHotEncoding", OneHotEncoder())])
preprocessing = ColumnTransformer([("Numerical Pipeline", num_pipeline, num_cols),
                                   ("Categorical Pipeline", cat_pipeline, cat_cols)])
x_train = preprocessing.fit_transform(x_train)
x_test = preprocessing.transform(x_test)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=50)
clf.fit(x_train, y_train)
from sklearn.metrics import r2_score
print(r2_score(y_test, clf.predict(x_test)))
from sklearn.ensemble import VotingRegressor
model = VotingRegressor([("LinearRegression", LinearRegression()), ("DecisionTree", DecisionTreeRegressor()), ("SVR", SVR())], verbose= 2)
model.fit(x_train, y_train)
print(r2_score(y_test, model.predict(x_test)))
