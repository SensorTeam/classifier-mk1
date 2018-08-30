"""
`python3 train.py -f path-to-training-data`
input training data csv
graph training data points
output saved model and transform
"""
from rgb_to_hsv import *
from config import *
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


########### main training function
def train(file):

	########### Check that macros are valid
	bit = [8, 14]
	cs = ["RGB", "HSV"]
	coords = ["polar", "cartesian"]
	weights = ["uniform", "distance"]
	if BIT in bit and COLORSPACE in cs and COORD_SYSTEM in coords and WEIGHT in weights:
		pass
	else:
		raise ValueError("Incorrect macros parsed. Check config.py")

	# open database
	with open(file, 'r') as f:
		reader = csv.reader(f)
		raw = list(reader)[:]
	data = np.asarray([row[1:] for row in raw])
	y = data[:, 0].astype(np.int)
	X = data[:, 1:].astype(np.float)


	############ RGB
	if COLORSPACE == "RGB":
		# Plot in 3D if data is rgb
		plt3D(X,y)
		# r/t, g/t, b/t
		if NORMALISED:
			final_X = normalise(X)		# 2D projected data
		# use og rgb values
		else:
			final_X = X 				# original 3D data

	############ HSV
	# 2D polar or cartesian
	else:
		# convert rgb to hsv with BIT = 14/8 
		x2d = []
		for item in X:
			r,g,b = item
			hsv = rgb_to_hsv(r,g,b,BIT)
			x2d.append([hsv[0],hsv[1]*100])

		# coordinate system conversion (polar or cartesian)
		if COORD_SYSTEM == "polar":
			xpol = []
			for item in x2d:
				pol = pol2cart(item[0], item[1])
				xpol.append(pol)
			final_X = xpol
		else:
			final_X = x2d
	

	########## Standardise data (mean=0, std=1)
	final_X = np.asarray(final_X)
	scaler = StandardScaler()
	scaler.fit(final_X)
	# Pickle the transform
	pickle.dump(scaler, open('scaler_transform.sav', 'wb'))
	learnset_data = scaler.transform(final_X)
	learnset_labels = y


	########## Create knn model
	model = neighbors.KNeighborsClassifier(N_NEIGHBOURS, weights=WEIGHT)
	model.fit(learnset_data, learnset_labels)
	pickle.dump(model, open('knn_model.sav', 'wb'))	# pickle it

	
	########## Plot in 2D showing class regions
	if NORMALISED or COLORSPACE == "HSV":
		h = 0.01  # step size in the mesh
		# Create color maps
		cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFF3AA', '#F3AAFF'])
		cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00', '#FFDB00', '#DC00FF'])

		# Plot the decision boundaries
		# Classify each point in the mesh [x_min, x_max]x[y_min, y_max].
		x_min, x_max = np.asarray(learnset_data)[:, 0].min() - 1, np.asarray(learnset_data)[:, 0].max() + 1
		y_min, y_max = np.asarray(learnset_data)[:, 1].min() - 1, np.asarray(learnset_data)[:, 1].max() + 1

		# Create mesh
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		# predict for each point in the mesh
		Z = model.predict(np.c_[xx.ravel(), yy.ravel()])		# column stack points

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.figure(figsize = (10,8))
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

		# Plot the training points
		plt.scatter(np.asarray(learnset_data)[:, 0], np.asarray(learnset_data)[:, 1], 
			c=y, cmap=cmap_bold, edgecolor='k', s=20)

		# Labels and titles
		if BIT == 14:
			imgtype = "RAW"
		else:
			imgtype = "JPG"
		plt.title("%s image filetype, %i bit, %s, k = %i, %s weighted" % 
			(imgtype, BIT, COLORSPACE, N_NEIGHBOURS, WEIGHT))
		if COLORSPACE == "HSV":
			plt.xlabel("hue")
			plt.ylabel("saturation")
		else:
			plt.xlabel("r-g")
			plt.ylabel("2b-r-g")
		plt.show()

	return


########### 3D plot
def plt3D(X, y):
	# Z scale standardise
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	data = X
	labels = y

	# Check that data is 3D
	if len(X[0])!= 3:
		raise ValueError('Data is not 3D RGB. Check config.py')
	# Plot points
	colours = ("r", "b", "g", "m", "y")
	x3d = []
	for iclass in range(N_CLASSES):
		x3d.append([[], [], []])
		for i in range(len(data)):
			if labels[i] == iclass:
				x3d[iclass][0].append(data[i][0])
				x3d[iclass][1].append(data[i][1])
				x3d[iclass][2].append(data[i][2])
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for iclass in range(N_CLASSES):
		ax.scatter(x3d[iclass][0], x3d[iclass][1], x3d[iclass][2], c=colours[iclass])
	plt.show()
	return

########### normalise rgb to r/t, g/b, b/t
def normalise(X):
	# normalise and put in x2d
	x2d = []
	# Project to (r-g, 2b-r-g) if using r/t, g/t, b/t data
	for entry in X:
		r,g,b = entry[0:3]
		tot = r+g+b
		r,g,b = r/tot, g/tot, b/tot
		x2d.append([1/math.sqrt(2)*(r-g), 1/math.sqrt(6)*(2*b-r-g)])
	return x2d


train("results_raw.csv")