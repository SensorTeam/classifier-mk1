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

data = np.array(['d73.jpg', 3, 0.358153146, 0.379450496, 0.262396358, 8.009993758])	# Filename, class, R, G, B, rIPD
data = np.array(['d78.jpg',	3, 0.369236403, 0.383311043, 0.247452554, 5.111111111])
data = np.array(['l172.jpg', 4, 0.38009686, 0.35891267, 0.26099047, 7.60504666])
data = np.array(['f110.jpg', 0, 0.407695133, 0.344030434, 0.248274433, 5.241833011])

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