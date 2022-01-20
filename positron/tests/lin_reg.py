import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import regression
import preprocessing as pre

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


np.random.seed(0)
data = load_boston()
df = pd.DataFrame(data=np.c_[data["data"], data["target"]], columns=list(data["feature_names"]) + ["TARGET"])

# sns.heatmap(df.corr(), annot=True)
# plt.scatter(df["LSTAT"], df["TARGET"], c="g", alpha=0.2, marker="o")
# plt.show()

train_set, test_set = pre.split_train_test(df)
xtrain, ytrain = pre.to_np(train_set["LSTAT"], train_set["TARGET"], single=True)
xtest, ytest = pre.to_np(test_set["LSTAT"], test_set["TARGET"], single=True)

"""
sns.set({"figure.figsize": (5, 7,), "grid.alpha": 0.0})
plt.title("Train set")
plt.xlabel("LSTAT")
plt.ylabel("TARGET")
plt.scatter(xtrain, ytrain, c="g", marker="o", alpha=0.5)
plt.show()
"""

plt.title("Test set")
plt.xlabel("LSTAT")
plt.ylabel("TARGET")
plt.scatter(xtest, ytest, c="b", marker="o", alpha=0.5)
plt.plot(xtest, regression.linear(xtrain, ytrain, xtest), c="r", alpha=1.0)
plt.legend(["regression.linear", "data"])
plt.show()
