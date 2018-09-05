BIT = 8 					# 8 or 14 (jpg or raw)

COLORSPACE = "HSV" 			# RGB or HSV
N_CLASSES = 5 				# number of classes

# RGB COLORSPACE
NORMALISED = True			# True gives R/t, G/t, B/t and 2D projection onto (r-g, 2b-r-g)
							# False is original rgb values classified in 3D

# FOR HSV COLORSPACE
COORD_SYSTEM = "polar"		# polar or cartesian

# Training
N_NEIGHBOURS = 8			# number of neighbours
WEIGHT = "uniform" 			# or "distance"

PATH_KNN_MODEL = "knn_model.sav"
PATH_SCALER_TRANSFORM = "scaler_transform.sav"
PATH_PLOT = "plot.sav"
