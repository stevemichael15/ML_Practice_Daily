import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = sns.load_dataset("tips")
# predict time
#EDA

#encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df.time = encoder.fit_transform(df["time"])
x = df.drop("time", axis = 1)
y = df["time"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.impute import SimpleImputer # for missing values
from sklearn.preprocessing import OneHotEncoder # for one hot encoding
from sklearn.preprocessing import StandardScaler # for scaling the data
from sklearn.pipeline import Pipeline # a sequence of data transformer
from sklearn.compose import ColumnTransformer # groups all the pipeline steps for each of the columns
cat_cols = [col for col in df.columns if df[col].dtypes == "category"]
num_cols = [col for col in df.columns if df[col].dtypes != "category"]
num_cols.remove("time")
  
#feature engineering using pipeline and columnTransformer
cat_pipeline = Pipeline(steps=[("Simple_Imputer", SimpleImputer(strategy="most_frequent")),("OneHot_Encoder", OneHotEncoder())])
num_pipeline = Pipeline(steps=[("Simple_Imputer", SimpleImputer(strategy="median")),("Standard_Scaler", StandardScaler())])
preprocessor = ColumnTransformer([("numerical_pipeline", num_pipeline, num_cols),
                                  ("categorical_pipeline", cat_pipeline, cat_cols)])
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[("LogisticRegression", LogisticRegression()), ("DecisionTreeClassifier", DecisionTreeClassifier()), ("SVC", SVC()), ("GaussianNB", GaussianNB())], voting= "hard")
model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, model.predict(x_test)))