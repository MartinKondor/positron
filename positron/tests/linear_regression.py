import sys
sys.path.insert(1, '../positron/positron')

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import calc
import datat
import algo


np.random.seed(0)
sns.set(rc={'figure.figsize': (15, 15,)})
plt.style.use('seaborn-white')

# Import a basic test dataset
df = pd.read_csv("positron/tests/btc_price.csv")
df.dropna(inplace=True)
print(df.head())

X = datat.date_to_stamp(np.array(df["date"]))
y = np.array(df["price"])

plt.figure(figsize=(7, 7))
plt.title('BTC Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.scatter(X, y, facecolors='None', edgecolors='k', alpha=.85)
plt.plot(X, algo.linear_regression(X, y), c="g")
plt.show()
