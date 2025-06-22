import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("Travel.csv")
# print(data.isnull().sum())
num_cols = [col for col in data.columns if data[col].dtypes != "object"]
cat_cols = [col for col in data.columns if data[col].dtypes == "object"]
# mean_values = data[num_cols].mean()
# data[num_cols] = data[num_cols].fillna(mean_values)
data["Gender"].replace(to_replace = {"Fe Male": "Female"}, inplace= True)
data = data.dropna()
# for cat_column in cat_cols:
#     sns.countplot(x = cat_column, data= data, palette= "viridis")
#     plt.figure(figsize=(8,6))
#     plt.title(f"Univariate Analysis: {cat_column}")
# plt.show()
# for num_column in num_cols:
#     sns.histplot(x = num_column, data= data, palette= "viridis")
#     plt.figure(figsize=(8,6))
#     plt.title(f"Univariate Analysis: {num_column}")
# plt.show()
data["TotalOfPersonVisiting"] = data["NumberOfPersonVisiting"]+data["NumberOfChildrenVisiting"]
data.drop("NumberOfChildrenVisiting", inplace= True, axis= 1)
data.drop("NumberOfPersonVisiting", inplace= True, axis= 1)
from sklearn.model_selection import train_test_split
x = data.drop("ProdTaken", axis= 1)
y = data["ProdTaken"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
cat_features = x.select_dtypes(include="object").columns
num_features = x.select_dtypes(exclude="object").columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler # both are data transformers
from sklearn.compose import ColumnTransformer #ColumnTransformer applies transformation to columns of an array or pandas dataframe
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop= "first")
preprocessor  = ColumnTransformer(
    [
        ('OneHotEncoder', oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features)
    ]
)
x_train = preprocessor.fit_transform(x_train)
x_train_dataframe = pd.DataFrame(x_train)
print(preprocessor.transform(x_test))
