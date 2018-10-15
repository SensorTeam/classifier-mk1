# split data into test and training data (1:1)
# open file
import random

file_j = open("data/realclose/real_close_jpg.csv", "r")
data_j = file_j.readlines()

test_j = open("data/realclose/real_close_jpg_test.csv", "a")
train_j = open("data/realclose/real_close_jpg_train.csv", "a")

file_r = open("data/realclose/real_close_raw.csv", "r")
data_r = file_r.readlines()

test_r = open("data/realclose/real_close_raw_test.csv", "a")
train_r = open("data/realclose/real_close_raw_train.csv", "a")

for i in range(1, len(data_r)):
	rand = random.randint(0,10)
	if rand > 7:
		test_r.write(data_r[i])
		test_j.write(data_j[i])
	else:
		train_r.write(data_r[i])
		train_j.write(data_j[i])

file_r.close()
file_j.close()
test_r.close()
test_j.close()
train_r.close()
train_j.close()
