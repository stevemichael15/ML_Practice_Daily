import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_regression
x, y = make_regression(n_samples=1000, n_features=2, n_targets=1, noise=3)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.svm import SVR
regressor = SVR(kernel="linear")
regressor.fit(x_train, y_train)
# sns.scatterplot(x= pd.DataFrame(x)[0], y=pd.DataFrame(x)[1], hue= y)
# plt.show()
