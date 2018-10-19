"""
`python3 train.py -f path-to-training-data`
input training data csv
graph training data points
output saved model and transform
"""

from rgb_to_hsv import *
import sys
sys.path.append('..')
from config import *
import numpy as np
import csv
import pickle
import math
# import matplotlib
# matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn import datasets, neighbors
from sklearn.preprocessing import StandardScaler
from config import *


########### main training function
def train(file, COLORSPACE, NORMALISED, COORD_SYSTEM, N_NEIGHBOURS, WEIGHT):

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
	pickle.dump(scaler, open(PATH_SCALER_TRANSFORM, 'wb'))
	learnset_data = scaler.transform(final_X)
	learnset_labels = y


	########## Create knn model
	model = neighbors.KNeighborsClassifier(N_NEIGHBOURS, weights=WEIGHT)
	model.fit(learnset_data, learnset_labels)
	pickle.dump(model, open(PATH_KNN_MODEL, 'wb'))	# pickle it

	
	########## Plot in 2D showing class regions
	if NORMALISED or COLORSPACE == "HSV":
		h = 0.01  # step size in the mesh
		# Create color maps
		light_colors = ['#FFAAAA', '#AAFFAA', '#FFF3AA', '#F3AAFF', '#AAAAFF']
		bold_colors = ['#FF0000', '#00FF00', '#FFDB00', '#DC00FF', '#0000FF']
		cmap_light = ListedColormap(light_colors)
		cmap_bold = ListedColormap(bold_colors)

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
		if COLORSPACE == "HSV":
			if COORD_SYSTEM == "polar":
				plt.xlabel("x")
				plt.ylabel("y")
			else:
				plt.xlabel("Hue")
				plt.ylabel("Saturation")
		else:
			plt.xlabel("r-g")
			plt.ylabel("2b-r-g")
		if BIT == 14:
			imgtype = "RAW"
		else:
			imgtype = "JPG"
		title = "%s image filetype, %i bit, %s polar, k = %i, %s weighted" % (imgtype, BIT, COLORSPACE, N_NEIGHBOURS, WEIGHT)
		ax = plt.title(title)
		

		legend_elements = [Line2D([0], [0], marker='o', color='w', label=CLASSES[0],
						markerfacecolor=bold_colors[0], markeredgecolor='k', markersize=5),
							Line2D([0], [0], marker='o', color='w', label=CLASSES[1],
						markerfacecolor=bold_colors[1], markeredgecolor='k', markersize=5),
							Line2D([0], [0], marker='o', color='w', label=CLASSES[2],
						markerfacecolor=bold_colors[2], markeredgecolor='k', markersize=5),
							Line2D([0], [0], marker='o', color='w', label=CLASSES[3],
						markerfacecolor=bold_colors[3], markeredgecolor='k', markersize=5),
							Line2D([0], [0], marker='o', color='w', label=CLASSES[4],
						markerfacecolor=bold_colors[4], markeredgecolor='k', markersize=5)]
		plt.legend(handles=legend_elements)
		pickle.dump(ax, open(PATH_PLOT, "wb"))
		plt.savefig(title+'.png', format='png', dpi=400)
		# plt.show()
		plt.close()

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
	colours = ("b", "g", "m", "r", "y")
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

	# plot all the points
	scatter_r = ax.scatter(x3d[0][0], x3d[0][1], x3d[0][2], c=colours[0])
	scatter_b = ax.scatter(x3d[1][0], x3d[1][1], x3d[1][2], c=colours[1])
	scatter_g = ax.scatter(x3d[2][0], x3d[2][1], x3d[2][2], c=colours[2])
	scatter_m = ax.scatter(x3d[3][0], x3d[3][1], x3d[3][2], c=colours[3])
	scatter_y = ax.scatter(x3d[4][0], x3d[4][1], x3d[4][2], c=colours[4])

	plt.legend([scatter_r, scatter_b, scatter_g, scatter_m, scatter_y], CLASSES, numpoints = 1)
	ax.set_xlabel('R (red)')
	ax.set_ylabel('G (green)')
	ax.set_zlabel('B (blue)')
	plt.title("RGB 3D, raw image filetype")
	# plt.show()
	plt.close()
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

# train('cow_sheep_data_raw.csv')

