import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df = df[df["target"] != 2]
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
# y_test = scaler.transform(y_test.values.reshape(-1, 1))
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
l1 = Lasso()
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# print(model.predict_proba(x_test)) # corresponding to higher probability class is predicted
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report, roc_curve, auc
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
ypred_proba = model.predict_proba(x_test)[:, -1]
# print(ypred_proba)
fpr, tpr, thresholds = roc_curve(y_test, ypred_proba)
roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(8,6))
# plt.plot(fpr, tpr, color="darkorange", linewidth= 2, label= 'ROC Curve(area = %0.2f)'% roc_auc)
# plt.plot([0,1], [0,1])
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve")
# plt.legend(loc ="lower right")
# plt.show()
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
kfold = KFold(n_splits= 5)
from sklearn.model_selection import cross_val_score
print(cross_val_score(model, x_train, y_train, cv= kfold, scoring="accuracy"))
