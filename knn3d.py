import numpy as np
import csv
from sklearn import datasets, neighbors
from sklearn.preprocessing import StandardScaler

NUMCLASS = 5 # number of classes
# How many nearest neighbours?
n_neighbors = 15
# 'uniform' or 'distance' # distance assigns weights proportional to the inverse of the distance from query point
WEIGHT = 'distance'

##################

# returns prediction accuracy for test data
def predict(filename, model, scaler):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		raw = list(reader)[1:]

	data = np.asarray([row[1:5] for row in raw])
	y = data[:, 0].astype(np.int)
	X = data[:, 1:].astype(np.float)

	# Standardise data (transform based on training data)
	X = scaler.transform(X)
	accuracy = 0
	for i in range(len(X)):
		actual = y[i]
		p = model.predict([X[i]])
		if actual == p:
			accuracy += 1
	accuracy = accuracy/len(X)
	return accuracy


# open database
with open('colourtrain.csv', 'r') as f:
	reader = csv.reader(f)
	raw = list(reader)[1:]

data = np.asarray([row[1:5] for row in raw])
y = data[:, 0].astype(np.int)
X = data[:, 1:].astype(np.float)

# Standardise data (mean=0, std=1)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
learnset_data = X
learnset_labels = y

# create knn model
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=WEIGHT)
clf.fit(X, y)

# plot in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
colours = ("r", "b")
X = []
for iclass in range(NUMCLASS):
    X.append([[], [], []])
    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(learnset_data[i][2])
colours = ("r", "g", "y", "b", "m")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(NUMCLASS):
       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()

# print predictions of test data
print("=======================\nk=%s, weights=%s" %(n_neighbors, WEIGHT))
print("Test Data Accuracy: " + str(predict("colourtest.csv", clf, scaler)))