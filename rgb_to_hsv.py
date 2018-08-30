import numpy as np

# converts polar coordinates to cartesian
def pol2cart(phi, r):
	x = r * np.cos(np.deg2rad(phi))
	y = r * np.sin(np.deg2rad(phi))
	return(x, y)

#This function converts RGB values to HSV. It takes the R, G, and B values
# as input, as well as the number corresponding to their bit (ie. 8-bit, 14-bit etc.)
# and outputs a list [HUE, SATURATION, VALUE].
def rgb_to_hsv(r, g, b, bit):

	#Scales the r, g, b values so that they are in the range 0-1.
	val = 2 ** bit - 1
	R = r / val
	G = g / val
	B = b / val

	#Finds the maximum and minimum r, g ,b values and calculates their difference.
	Cmax = max(R , G , B)
	Cmin = min(R , G , B)
	delta = Cmax - Cmin

	#HUE. Calculates the hue.
	if delta == 0:
		H = 0

	elif Cmax == R:
		H = 60 * (((G - B)/delta) % 6)

	elif Cmax == G:
		H = 60 * (((B - R)/delta) + 2)

	elif Cmax == B:
		H = 60 * (((R - G)/delta) + 4)

	#SATURATION. Calculates the saturation.
	if Cmax == 0:
		S = 0

	elif Cmax != 0:
		S = delta / Cmax

	#VALUE. Calculates the value.
	V = Cmax

	return [H, S, V]

#TESTING.
"""r, g, b = 1805.75671,1950.30186,1703.098219
bit = 14
hsvlist = hsv_to_rgb(r, g, b, bit)
print(hsvlist)"""