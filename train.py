"""
`python3 train.py -f path-to-training-data`
input training data csv
graph training data points
output saved model and transform
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import argparse
import pickle
from sklearn import datasets, neighbors
from sklearn.preprocessing import StandardScaler


NUMDIM = 4 # number of dimensions/variables
NUMCLASS = 5 # number of classes
# How many nearest neighbours?
n_neighbors = 15
# 'uniform' or 'distance' # distance assigns weights proportional to the inverse of the distance from query point
WEIGHT = 'distance'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help = "path to the training data csv file")
args = vars(ap.parse_args())

file = args["file"]

# open database
with open(file, 'r') as f:
	reader = csv.reader(f)
	raw = list(reader)[1:]

data = np.asarray([row[1:NUMDIM+2] for row in raw])
y = data[:, 0].astype(np.int)
X = data[:, 1:].astype(np.float)

# Standardise data (mean=0, std=1)
scaler = StandardScaler()
scaler.fit(X)
# pickle the transform
pickle.dump(scaler, open('t/scaler_transform.sav', 'wb'))
X = scaler.transform(X)
learnset_data = X
learnset_labels = y

# create knn model
model = neighbors.KNeighborsClassifier(n_neighbors, weights=WEIGHT)
model.fit(X, y)
# pickle it
pickle.dump(model, open('t/knn_model.sav', 'wb'))

# plot in 3D
X = []
colours = ("r", "g", "y", "b", "m")
for iclass in range(NUMCLASS):
    X.append([[], [], []])
    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(learnset_data[i][2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(NUMCLASS):
       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()