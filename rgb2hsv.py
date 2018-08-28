import csv
import colorsys
import numpy as np
from rgb_to_hsv import *

file = "results_raw.csv"
mode = "raw"		# raw or jpg?

# open database
with open(file, 'r') as f:
	reader = csv.reader(f)
	raw = list(reader)[:]

data = np.asarray([row[1:5] for row in raw])
y = data[:, 0].astype(np.int)
X = data[:, 1:].astype(np.float)

for item in X:
	r,g,b = item[0], item[1], item[2]
	if mode == "raw":
		hsv = rgb_to_hsv(r,g,b,14)
		print(hsv[0], hsv[1])
	elif mode == "jpg":
		hsv = rgb_to_hsv(r,g,b,8)
		print(hsv[0], hsv[1])