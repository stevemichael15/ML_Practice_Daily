import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=1000, n_features= 10,n_redundant= 5, n_informative= 5, n_classes=2, random_state=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc
proba = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, proba)
roc_auc = auc(fpr,tpr)
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
# by defualt cut off is 0.5 to decide whether the predicted dp is from class 0 or 1
# to make customized cut off
from sklearn.metrics import precision_score, recall_score, accuracy_score
thresholds1 = np.linspace(0,1, 100)
precisions = []
recalls = []
accuracies = []
for threshold in thresholds1:
    y_pred_threshold = (proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    accuracy = accuracy_score(y_test, y_pred_threshold)
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
# plt.figure(figsize=(10,6))
# plt.plot(thresholds1, precisions, label="Precision")
# plt.plot(thresholds1, recalls, label= "Recall")
# plt.plot(thresholds1, accuracies, label = "Accuracy")
# plt.xlabel("Threshold Probability")
# plt.ylabel("Score")
# plt.title("Precision, Recall and Accuracy vs. Threshold Proability")
# plt.legend()
# plt.grid(True)
# plt.show()
# 0.4 is the optimal threshold or the cut off
new_pred_levels = np.where(proba>0.4, 1, 0)
print(accuracy_score(y_test, new_pred_levels))
from sklearn.model_selection import KFold
cv = KFold(n_splits= 5)
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(model, x_train, y_train, cv= cv)
print(np.mean(cv_score))