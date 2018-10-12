# split data into test and training data (1:1)
# open file
import random

file = open("data/modelremoteadj/38model_remote_jpg.csv", "r")
data = file.readlines()

test = open("data/modelremoteadj/38model_remote_jpg_test.csv", "a")
train = open("data/modelremoteadj/38model_remote_jpg_train.csv", "a")

for i in range(1, len(data)):
	rand = random.randint(0,10)
	if rand > 7:
		test.write(data[i])
	else:
		train.write(data[i])

file.close()
test.close()
train.close()
