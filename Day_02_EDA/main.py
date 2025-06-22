import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("BIKE DETAILS.csv")
print(f"The range of the price is : {df['selling_price'].max()-df['selling_price'].min()}") # for range
print(f"The median of the price is {df['selling_price'].median()}")
print(f"Most common seller type model is {df['name'].mode().tolist()}")
print(f"Bikes that have driven more than 50,000kms are: {df[df['km_driven']>50000]['name'].tolist()}")
print(f"Bikes average km driven  are: {df.groupby('owner')['km_driven'].mean()}")
print(f"Bikes that are from the year 2015 or older are {df[df['year']>=2015]['name'].tolist()}")
print(f"Missing values across the dataset: {df.isna().sum()}")
print(f"Highest ex_showroom_price recorded is: {df['ex_showroom_price'].max()} and the name of the bike is: {df[df['ex_showroom_price'] == df['ex_showroom_price'].max()]['name']}")
print(f"The total number of bikes listed by each seller type is: {df['seller_type'].value_counts()}")
km_driven_1st = df[df["owner"] =="1st owner"]["km_driven"]
selling_price_1st = df[df["owner"] =="1st owner"]["selling_price"]
plt.scatter(km_driven_1st, selling_price_1st)
plt.xlabel("km_driven_1st")
plt.ylabel("selling_price_1st")
plt.show()
df.drop("ex_showroom_price", axis= 1, inplace= True)

median_value = df["km_driven"].median()
df["km_driven"] = df["km_driven"].apply(lambda x: df["km_driven"].median() if (x < Q1 - 1.5 * IQR or x > Q3 + 1.5 * IQR) else x)
df.loc[(df["km_driven"] >= upper_bound) | (df["km_driven"] <= lower_bound), "km_driven"] = median_value

def treating_outliers(col):
    Q3 = np.quantile(col, 0.75)
    Q1 = np.quantile(col, 0.25)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median_value = col.median()
    for i in range(0, len(col)):
        if (col[i] >= upper_bound) or (col[i] <= lower_bound):
            col[i] = median_value
    return col
def identifying_outliers(col):
    outliers = []
    Q3 = np.quantile(col, 0.75)
    Q1 = np.quantile(col, 0.25)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median_value = col.median()
    for i in range(0, len(col)):
        if (col[i] >= upper_bound) or (col[i] <= lower_bound):
            outliers.append(i)
    return outliers
# df["ex_showroom_price"] = df["ex_showroom_price"].astype(int)
df["km_driven"] = treating_outliers(df["km_driven"])
df["km_driven"] = treating_outliers(df["km_driven"])
df["km_driven"] = treating_outliers(df["km_driven"])
df["year"] = treating_outliers(df["year"])
df["year"] = treating_outliers(df["year"])
df["selling_price"] = treating_outliers(df["selling_price"])
df["selling_price"] = treating_outliers(df["selling_price"])
df["selling_price"] = treating_outliers(df["selling_price"])
df["selling_price"] = treating_outliers(df["selling_price"])
df["selling_price"] = treating_outliers(df["selling_price"])
# for col in df.columns:
#     if df[col].dtypes !="object":
#         print(f"{col}: {len(identifying_outliers(df[col]))}:{identifying_outliers(df[col])}")
df["owner"].replace(to_replace={"1st owner": 1, "2nd owner": 2, "3rd owner": 3, "4th owner": 4}, inplace=True)
df["seller_type"].replace(to_replace={"Individual": 1, "Dealer": 2}, inplace= True)
x = df.drop("selling_price", axis = 1)
y = df["selling_price"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
cat_features = x.select_dtypes(include="object").columns
num_features = x.select_dtypes(exclude="object").columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
[
        ('OneHotEncoder', oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features)
    ]
)
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler.transform(y_test.values.reshape(-1, 1))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
coefficient = model.coef_
intercept = model.intercept_
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
print(f"The r2 score of the model is: {r2_score(y_test, y_pred)}")
# x1 = df.iloc[:, :-1]
# y1 = df.iloc[:, -1]
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
elastic_net = ElasticNet()
grid_model = GridSearchCV(estimator=elastic_net, param_grid={"alpha": [0.1, 00.1, 20, 100]}, verbose=2, cv=5)
grid_model.fit(x_train, y_train)
grid_model.predict(x_test)
print(f"The best hyperparameter is: {grid_model.best_params_}")
print(f"The best r2 score through grid search cv is: {r2_score(y_test, grid_model.predict(x_test))}")