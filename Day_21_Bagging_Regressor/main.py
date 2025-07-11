import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
lr_clf = LogisticRegression()
dt_clf = DecisionTreeClassifier()
sv_clf = SVC()
gn_clf = GaussianNB()
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[("Logistic", lr_clf), ("Decision_Tree", dt_clf), ("SVM", sv_clf), ("Naive_Bayes", gn_clf)], voting= "hard")
model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, model.predict(x_test)))
