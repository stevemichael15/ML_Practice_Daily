import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=2, n_redundant=0, random_state=1)
# sns.scatterplot(x = pd.DataFrame(x)[0], y = pd.DataFrame(x)[1], hue=y)
# plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.svm import SVC
classifier = SVC(kernel= "linear")
classifier.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, classifier.predict(x_test)))
from sklearn.model_selection import GridSearchCV
params = {
    "C" : [0.1, 0.2, 1, 2, 3, 10, 50, 100],
    "gamma" : [1, 0.1, 0.2, 0.001, 0.003],
    "kernel" : ["linear"]
}
model = GridSearchCV(SVC(), param_grid=params, cv=5, verbose= 2)
model.fit(x_train, y_train)
print(model.best_params_)
classifier = SVC(kernel= "linear", C= 0.1, gamma= 1)
classifier.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, classifier.predict(x_test)))