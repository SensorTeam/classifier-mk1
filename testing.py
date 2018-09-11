from classify import *
from train import *
import csv

train("data_jpg_train.csv")

with open("data_jpg_test.csv", 'r') as f:
	reader = csv.reader(f)
	data = list(reader)[:]

correct, count = 0, 0
for entry in data:
	_, p = classify(entry)
	if p == int(entry[1]):
		correct+=1
	count+=1

accuracy = float(correct/count)
print("{:0.4f}".format(accuracy))

