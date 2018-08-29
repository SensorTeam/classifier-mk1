"""
`python3 train.py -f path-to-training-data`
input training data csv
graph training data points
output saved model and transform
"""

import numpy as np
import csv
import argparse
import pickle
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors
from sklearn.preprocessing import StandardScaler


NUMDIM = 2 # number of dimensions/variables
NUMCLASS = 5 # number of classes
# How many nearest neighbours?
n_neighbors = 10
# 'uniform' or 'distance' # distance assigns weights proportional to the inverse of the distance from query point
WEIGHT = 'uniform'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help = "path to the training data csv file")
args = vars(ap.parse_args())

file = args["file"]

# open database
with open(file, 'r') as f:
	reader = csv.reader(f)
	raw = list(reader)[:]

data = np.asarray([row[1:NUMDIM+2] for row in raw])
y = data[:, 0].astype(np.int)
X = data[:, 1:].astype(np.float)

# Standardise data (mean=0, std=1)
scaler = StandardScaler()
scaler.fit(X)
# pickle the transform
pickle.dump(scaler, open('scaler_transform.sav', 'wb'))
X = scaler.transform(X)
learnset_data = X
learnset_labels = y


# # PLOT POINTS IN 3D
# colours = ("r", "b", "g", "m", "y")
# x3d = []
# for iclass in range(NUMCLASS):
# 	x3d.append([[], [], []])
# 	for i in range(len(learnset_data)):
# 		if learnset_labels[i] == iclass:
# 			x3d[iclass][0].append(learnset_data[i][0])
# 			x3d[iclass][1].append(learnset_data[i][1])
# 			x3d[iclass][2].append(learnset_data[i][2])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for iclass in range(NUMCLASS):
# 	ax.scatter(x3d[iclass][0], x3d[iclass][1], x3d[iclass][2], c=colours[iclass])
# plt.show()


# WORK IN 2D
# Project to (r-g, 2b-r-g) if using r/t, g/t, b/t data
# 
x2d = []
for entry in learnset_data:
	# r,g,b = entry[0:3]
	# x2d.append([1/math.sqrt(2)*(r-g), 1/math.sqrt(6)*(2*b-r-g)])
	x2d.append(entry)		# for data that is already 2D
# Create 2D knn model
model = neighbors.KNeighborsClassifier(n_neighbors, weights=WEIGHT)
model.fit(x2d, y)
pickle.dump(model, open('knn_model.sav', 'wb'))	# pickle it


# PLOT IN 2D WITH REGIONS
h = 0.01  # step size in the mesh
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFF3AA', '#F3AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00', '#FFDB00', '#DC00FF'])

# Plot the decision boundaries
# classify each point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = np.asarray(x2d)[:, 0].min() - 1, np.asarray(x2d)[:, 0].max() + 1
y_min, y_max = np.asarray(x2d)[:, 1].min() - 1, np.asarray(x2d)[:, 1].max() + 1

# create mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# predict for each point in the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])		# column stack points

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize = (10,8))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(np.asarray(x2d)[:, 0], np.asarray(x2d)[:, 1], 
	c=y, cmap=cmap_bold, edgecolor='k', s=20)

# Labels and titles
plt.title("k = %i, weights = '%s'" % (n_neighbors, WEIGHT))
plt.xlabel("hue")
plt.ylabel("saturation")
plt.show()
