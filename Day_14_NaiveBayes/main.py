import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
x = load_iris().data
y = load_iris().target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.naive_bayes import GaussianNB # features are continuous in nature
clf = GaussianNB()
clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, clf.predict(x_test)))
