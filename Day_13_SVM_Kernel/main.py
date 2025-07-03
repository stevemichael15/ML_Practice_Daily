import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification
x, y = make_classification(1000, 3, n_classes=2, random_state=1, n_redundant= 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
model = SVC(C= 2, gamma= "auto", kernel= "rbf")
model.fit(x_train, y_train)
print(accuracy_score(y_test, model.predict(x_test)))
# from sklearn.model_selection import GridSearchCV
# params = {
#     "kernel" : ["linear", "rbf", "sigmoid", "poly"],
#     "gamma" : ["auto", "scale"],
#     "C" :  [0, 2, 0.1, 0.2, 0.001]
# }
# model = GridSearchCV(SVC(), param_grid=params, cv= 5, verbose=2)
# model.fit(x_train,y_train)
# print(model.best_params_)
model = SVC(C= 2, gamma= "auto", kernel= "rbf")
model.fit(x_train, y_train)
print(accuracy_score(y_test, model.predict(x_test)))
