import os
import sys

from sklearn import preprocessing
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import classification as cls
import preprocessing as pre

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


np.random.seed(0)
data = load_iris()
df = pd.DataFrame(data=np.c_[data["data"], data["target"]], columns=list(data["feature_names"]) + ["TARGET"])
df.dropna(inplace=True)

# Drop a class to make it binary classification
index_names = df[df['TARGET'] == 1].index
df.drop(index_names, inplace=True)

#plt.scatter(df["petal width (cm)"], df["petal length (cm)"], c=df["TARGET"], marker="o")
#plt.show()

train_set, test_set = pre.split_train_test(df)
xtrain = np.c_[pre.to_np(train_set["petal width (cm)"], train_set["petal length (cm)"], single=True)]
ytrain = np.array(pre.to_np(test_set["TARGET"], single=True))

# Prepare classes
ytrain[ytrain == 2] = 1
knn = cls.KNearestNeighbor(xtrain, ytrain, k=2)
