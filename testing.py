from classify import *
from train import *
import csv

# train("data/modelremote/model_remote_raw_train.csv")

# with open("data/modelremote/model_remote_raw_test.csv", 'r') as f:
# 	reader = csv.reader(f)
# 	data = list(reader)[:]

# correct, count = 0, 0
# for entry in data:
# 	_, p = classify(entry)
# 	if p == int(entry[1]):
# 		correct+=1
# 	count+=1

# accuracy = float(correct/count)
# print("{:0.4f}".format(accuracy))

##################################################################

combos = [	
			# ["RGB", False, "polar", 3, "uniform"],
			# ["RGB", False, "polar", 3, "distance"],
			# ["RGB", False, "polar", 6, "uniform"],
			# ["RGB", False, "polar", 6, "distance"],
			# ["RGB", False, "polar", 9, "uniform"],
			# ["RGB", False, "polar", 9, "distance"],
			# ["RGB", False, "polar", 12, "uniform"],
			# ["RGB", False, "polar", 12, "distance"]

			# ["RGB", True, "polar", 3, "uniform"],
			# ["RGB", True, "polar", 3, "distance"],
			# ["RGB", True, "polar", 6, "uniform"],
			# ["RGB", True, "polar", 6, "distance"],
			# ["RGB", True, "polar", 9, "uniform"],
			# ["RGB", True, "polar", 9, "distance"],
			# ["RGB", True, "polar", 12, "uniform"],
			# ["RGB", True, "polar", 12, "distance"]

			# ["HSV", True, "cartesian", 3, "uniform"],
			# ["HSV", True, "cartesian", 3, "distance"],
			# ["HSV", True, "cartesian", 6, "uniform"],
			# ["HSV", True, "cartesian", 6, "distance"],
			# ["HSV", True, "cartesian", 9, "uniform"],
			# ["HSV", True, "cartesian", 9, "distance"],
			# ["HSV", True, "cartesian", 12, "uniform"],
			# ["HSV", True, "cartesian", 12, "distance"]

			# ["HSV", True, "polar", 3, "uniform"],
			# ["HSV", True, "polar", 3, "distance"],
			# ["HSV", True, "polar", 6, "uniform"],
			# ["HSV", True, "polar", 6, "distance"],
			# ["HSV", True, "polar", 9, "uniform"],
			# ["HSV", True, "polar", 9, "distance"],
			# ["HSV", True, "polar", 12, "uniform"],
			# ["HSV", True, "polar", 12, "distance"]
		]

for combo in combos:	
	print(combo)
	[COLORSPACE, NORMALISED, COORD_SYSTEM, N_NEIGHBOURS, WEIGHT] = combo

	train("data/modelremoteadj/38model_remote_raw_train.csv", COLORSPACE, NORMALISED, COORD_SYSTEM, N_NEIGHBOURS, WEIGHT)

	with open("data/modelremoteadj/38model_remote_raw_test.csv", 'r') as f:
		reader = csv.reader(f)
		data = list(reader)[:]

	correct, count = 0, 0
	for entry in data:
		_, p = classify(entry, COLORSPACE, NORMALISED, COORD_SYSTEM, N_NEIGHBOURS, WEIGHT)
		if p == int(entry[1]):
			correct+=1
		count+=1

	accuracy = float(correct/count)
	print("{:0.4f}".format(accuracy))




