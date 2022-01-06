import sys
sys.path.insert(1, '../positron/positron')

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import calc
import datat
import regression


np.random.seed(0)
sns.set(rc={'figure.figsize': (7, 7,)})
plt.style.use('seaborn-white')

df = pd.read_csv("positron/tests/bin_classes.csv")
df.dropna(inplace=True)
print(df.head())

X1 = np.array(df["x1"])
X2 = np.array(df["x2"])
y = np.array(df["y"])

r = regression.logistic(y.T * X1) + regression.logistic(y.T * X2)
r /= 2
r[r > 0.5] = 1
r[r < 0.5] = 0
r[r == 0.5] = 0
print(r)

plt.title('Sample data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(X1, X2, facecolors='None', edgecolors='k', alpha=.85, c=y)
plt.show()

plt.title('Predictions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(X1, X2, facecolors='None', edgecolors='k', alpha=.85, c=r)
plt.show()
