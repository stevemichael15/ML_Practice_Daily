import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = sns.load_dataset("tips")
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df["time"] = label.fit_transform(df["time"])
x = df.drop("time", axis = 1)
y = df["time"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
cat_cols = [col for col in df.columns if df[col].dtypes == "category"]
num_cols = [col for col in df.columns if df[col].dtypes != "category"]
num_cols.remove("time")
num_pipeline = Pipeline([("SimpleImputer", SimpleImputer(strategy="mean")), ("StandardScaler", StandardScaler())])
cat_pipeline = Pipeline([("SimpleImputer", SimpleImputer(strategy="most_frequent")), ("One Hot Encoding", OneHotEncoder())])
preprocessor = ColumnTransformer([("numerical pipeline", num_pipeline, num_cols),
                   ("categorical_pipeline", cat_pipeline, cat_cols)])
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# params = {
#     "max_depth" : [1, 2, 3, 4, 5, None],
#     "n_estimators" : [30, 40, 50, 100, 200, 300],
#     "criterion" : ["gini", "entropy"]
# }
# clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions= params, verbose = 2, cv= 5, scoring=  "accuracy", n_iter=10)
# clf.fit(x_train, y_train)
model = RandomForestClassifier(n_estimators= 50, max_depth= 2, criterion= "gini")
model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, model.predict(x_test)))