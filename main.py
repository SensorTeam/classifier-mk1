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

data = np.array(['d73.jpg', 3, 0.358153146, 0.379450496, 0.262396358])	# Filename, class, R, G, B, rIPD

def main(data):
	# append to database
	f = open('test_database.csv', 'a')
	writer = csv.writer(f)
	writer.writerow(data)
	f.close()

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

# return prediction for one data
def predict(data, model, scaler):
	X = data[2:].astype(np.float)
	# Standardise data (transform based on training data)
	X = X.reshape(1,-1)
	X = scaler.transform(X)
	p = model.predict(X)
	return p[0]

main(data)