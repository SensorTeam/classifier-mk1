import csv
import colorsys
import numpy as np
from rgb_to_hsv import *

file = "results_raw.csv"
bit = 14		# 8 for jpg, 14 for raw

def pol2cart(r, phi):
	x = r * np.cos(np.deg2rad(phi))
	y = r * np.sin(np.deg2rad(phi))
	return(x, y)

# open database
with open(file, 'r') as f:
	reader = csv.reader(f)
	raw = list(reader)[:]

data = np.asarray([row[0:5] for row in raw])
filenames = data[:, 0]
y = data[:, 1].astype(np.int)
X = data[:, 2:].astype(np.float)

f = open(file[:-4] + "pol.csv", 'w')
writer = csv.writer(f)

for i in range(len(X)):
	r,g,b = X[i]
	hsv = rgb_to_hsv(r,g,b,bit)
	pol = pol2cart(hsv[1]*100, hsv[0])
	writer.writerow([filenames[i], y[i], pol[0], pol[1]])
	
f.close()