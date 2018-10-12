"""
# input array of data
# append/store it the database
# load model
# predict using model
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
from rgb_to_hsv import *
import sys
sys.path.append('..')
from config import *

# Classifies based on pre-existing loaded training set
# Filename, class, R, G, B
def classify(data, COLORSPACE, NORMALISED, COORD_SYSTEM, N_NEIGHBOURS, WEIGHT):
	# load model and transform
	model = pickle.load(open(PATH_KNN_MODEL, 'rb'))
	scaler = pickle.load(open(PATH_SCALER_TRANSFORM, 'rb'))
	# print("\n=================================================")
	# print("LOADED DATA = %s"% data)
	
	# label?
	try:
		c = int(data[1])
	except:
		c = 'None'

	########### Check that macros are valid
	bit = [8, 14]
	cs = ["RGB", "HSV"]
	coords = ["polar", "cartesian"]
	weights = ["uniform", "distance"]

	if BIT in bit and COLORSPACE in cs and COORD_SYSTEM in coords and WEIGHT in weights:
		pass
	else:
		raise ValueError("Incorrect macros parsed. Check config.py")

	# transform the data using desired method
	transformed_data = transform(np.asarray(data), COLORSPACE, NORMALISED, COORD_SYSTEM, N_NEIGHBOURS, WEIGHT)

	# standardise using scaler
	newX = np.asarray(transformed_data)
	# Standardise data (transform based on training data)
	newX = newX.reshape(1,-1)
	newX = scaler.transform(newX)

	# If 2D, plot the new point in classification space
	if NORMALISED or COLORSPACE == "HSV":
		ax = pickle.load(open(PATH_PLOT, "rb"))
		plt.scatter(newX[0][0], newX[0][1], s=250,marker='*', facecolors='w', edgecolors='k',linewidths=1)
		# plt.show()
		plt.close()

	# make prediction
	p = model.predict(newX)[0]
	neighbour_dist = model.kneighbors(newX)[0]
	
	# RETURN RESULTS
	returnstr = "==========================\nTransformed data: " +str(transformed_data)
	returnstr += "\nFILE: %s\nLABEL: %s\n"%(data[0],str(c))
	
	# if closest neighbour is too far away, possibly new class
	if neighbour_dist[0][0] > 1:
		returnstr += "Distance from closest neighbour: %f\n" % neighbour_dist[0][0]
		returnstr += "Class unclear\n"
	else:
		returnstr += "PREDICTION: %i\n"%(p)
	return returnstr, p


# transform data using desired method from flags
def transform(data, COLORSPACE, NORMALISED, COORD_SYSTEM, N_NEIGHBOURS, WEIGHT):
	X = data[2:].astype(np.float)

	# data is always given in original rgb values
	r,g,b = X

	# normalise for 2D RGB
	if COLORSPACE == "RGB":
		if NORMALISED:			# 2D r/t, g/t, b/t
			return normalise(r,g,b)
		else:
			return [r,g,b]

	# transform to 2D hue and saturation
	else:
		hsv = rgb_to_hsv(r,g,b, BIT)
		if COORD_SYSTEM == "polar":		# polar coordinates
			pol = pol2cart(hsv[0], hsv[1]*100)
			return pol
		else:		# cartesian
			return [hsv[0], hsv[1]*100]
	

def normalise(r,g,b):
	tot = r+g+b
	r,g,b = r/tot, g/tot, b/tot
	x = 1/math.sqrt(2)*(r-g)
	y = 1/math.sqrt(6)*(2*b-r-g)
	return [x,y]

# print(classify(["images/model_remote/IMG_8278",2,1305.666667,1504.9375,1188.75])[0])
# print(classify(["images/model_remote/IMG_8372",4,1495.214022,2401.363303,2308.036496])[0])
# print(classify(["images/model_remote/IMG_8297",3,1124.382716,1080.037072,993.8296296])[0])
# print(classify(["images/model_remote/IMG_8321",3,1386.930233,1369.709302,1502.911111])[0])




# print(classify(["images/model_remote/IMG_8353",4,2950.442581,6101.878283,6543.457801])[0])
# print(classify(["images/model_remote/IMG_8355",4,2099.285057,4216.685315,4114.765808])[0])
# print(classify(["images/model_remote/IMG_8357",4,2186.653488,4343.09633,4246.896313])[0])
# print(classify(["images/model_remote/IMG_8358",4,2058.188192,3978.80948,3800.027624])[0])
# print(classify(["images/model_remote/IMG_8359",4,2101.098837,4100.709709,3932.389105])[0])
# print(classify(["images/model_remote/IMG_8360",4,1665.134199,2901.53276,2738.946809])[0])
# print(classify(["images/model_remote/IMG_8362",4,1544.830588,2562.172695,2442.331019])[0])
# print(classify(["images/model_remote/IMG_8366",4,1597.663812,2715.600214,2573.590129])[0])
# print(classify(["images/model_remote/IMG_8370",4,1447.49004,2287.208661,2236.555556])[0])
# print(classify(["images/model_remote/IMG_8371",4,1328.257235,1944.346216,1898.506369])[0])
# print(classify(["images/model_remote/IMG_8372",4,1495.214022,2401.363303,2308.036496])[0])
# print(classify(["images/model_remote/IMG_8372",4,467.1286784,737.2059691,721.5324291])[0])
# print(classify(["images/model_remote/IMG_8376",4,1483.75,2290.928025,2168.187879])[0])
# print(classify(["images/model_remote/IMG_8378",4,1448.577381,2345.546828,2304.487805])[0])
# print(classify(["images/model_remote/IMG_8380",4,1503.31,2521.718447,2493.36])[0])
# print(classify(["images/model_remote/IMG_8382",4,1527.463636,2545.738938,2467.631579])[0])
# print(classify(["images/model_remote/IMG_8384",4,901.6017386,1287.749243,1030.914196])[0])
# print(classify(["images/model_remote/IMG_8387",4,555.9770079,572.1700523,498.9143373])[0])
# print(classify(["images/model_remote/IMG_8390",4,1304.149425,1891.9,1847.197674])[0])
# print(classify(["images/model_remote/IMG_8394",4,1189.396694,1616.795918,1659.528926])[0])
# print(classify(["images/model_remote/IMG_8396",4,745.7272308,890.9962449,819.7685978])[0])
# print(classify(["images/model_remote/IMG_8403",4,1192.540541,1654.453333,1682.897436])[0])


