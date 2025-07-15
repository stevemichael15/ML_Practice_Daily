import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
x, y =make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1)
clf = XGBClassifier()
clf.fit(x_train, y_train)
print(confusion_matrix(y_test, clf.predict(x_test)))
print(accuracy_score(y_test, clf.predict(x_test)))
print(classification_report(y_test, clf.predict(x_test)))
from sklearn.model_selection import GridSearchCV
# params = {
#     "n_estimators": [100, 200, 300],
#     "learning_rate": [0.01, 0.1, 1, 1.5, 2],
# }
# model = GridSearchCV(GradientBoostingClassifier(), param_grid=params, cv=5, verbose=2)
# model.fit(x_train, y_train)
# print(accuracy_score(y_test, model.best_estimator_.predict(x_test)))

# -------------------------------XG_Boost-------------------------------
# from sklearn.datasets import make_regression
# x, y = make_regression(n_samples=1000, n_features=2, random_state=1, noise=10)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# reg = XGBRegressor()
# reg.fit(x_train, y_train)
# from sklearn.metrics import r2_score
# print(r2_score(y_test, reg.predict(x_test)))
