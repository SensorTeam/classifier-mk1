"""
`python3 main.py`
# input array of data
# append/store it the database
# load model
# predict using model
"""

import pickle
import csv
import numpy as np
from rgb_to_hsv import *


# return prediction for one data
def predict(data, model, scaler):
	X = data[2:].astype(np.float)
	# transform to 2D polar hue and saturation
	r,g,b = X
	hsv = rgb_to_hsv(r,g,b,14)		# bit = 14 if raw, 8 if jpg
	pol = pol2cart(hsv[1]*100, hsv[0])
	newX = np.asarray(pol)
	# Standardise data (transform based on training data)
	newX = newX.reshape(1,-1)
	newX = scaler.transform(newX)
	p = model.predict(newX)
	return p[0]


# Classifies based on pre-existing loaded training set
# Filename, class, R, G, B, rIPD
def classify(data):
	# # append to database
	# f = open('test_database.csv', 'a')
	# writer = csv.writer(f)
	# writer.writerow(data)
	# f.close()

	# load model and transform
	model = pickle.load(open('knn_model.sav', 'rb'))
	scaler = pickle.load(open('scaler_transform.sav', 'rb'))

	# label?
	try:
		c = data[1].astype(np.int)
	except:
		c = 'None'

	# make prediction
	print('File: ' + data[0])
	print('Label: ' + str(c))
	print('Prediction: ' + str(predict(data, model, scaler)))
	return
