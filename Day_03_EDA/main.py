import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = sns.load_dataset("titanic")
df["age"] = df["age"].fillna(df["age"].mean())
df.dropna(subset=["embark_town"], inplace=True)
print(df.info())
discrete_col = [col for col in df.columns if len(df[col].unique()) <= 5]
for col in discrete_col:
    print(f"{col}:{df[col].unique()}")
sns.boxplot(x= df["survived"], y=df["age"])
df.embark_town.value_counts().plot(kind= "pie", autopct= "%1.1f%%")
plt.show()

#To know how many survived and how many not from each town
print(df.groupby(["survived", "embark_town"]).size())
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='embark_town', hue='survived')
plt.title('Survival Count by Embarkation Town')
plt.xlabel('Embarkation Town')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()
