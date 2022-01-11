import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


sns.set(rc={'figure.figsize': (8, 5,)})
plt.style.use('seaborn-white')

df = pd.read_csv("housing.csv")
df.dropna(inplace=True)
print(df.head())

#sns.heatmap(df.corr(), annot=True)

X = df["median_income"]
y = df["median_house_value"]

plt.show()
