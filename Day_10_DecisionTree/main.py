import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=1000, n_features= 20, n_informative=5, n_redundant=5, n_classes=4)
from sklearn.model_selection import train_test_split
x_train, x_test,  y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
params = {
    "criterion" : ["gini", "entropy"],
    "splitter" : ["best", "random"],
    "max_depth" : [4, 5, 6, 7],
    "max_features" : ["sqrt", "log2"]
}
model = GridSearchCV(DecisionTreeClassifier(), param_grid= params, scoring="accuracy", cv = 5, verbose= 2)
model.fit(x_train, y_train)
print(model.best_params_)
model2 = DecisionTreeClassifier(criterion="entropy", max_depth= 6, max_features= "log2", splitter= "best")
model2.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, model2.predict(x_test)))