# split data into test and training data (1:1)
# open file
import random

file = open("data/real/dead_raw.csv", "r")
data = file.readlines()

test = open("data/real/dead_raw_test_split2.csv", "a")
train = open("data/real/dead_raw_train_split2.csv", "a")

for i in range(1, len(data)):
	rand = random.randint(0,10)
	if rand > 7:
		test.write(data[i])
	else:
		train.write(data[i])

file.close()
test.close()
train.close()
