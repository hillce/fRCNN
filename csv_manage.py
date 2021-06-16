import csv
import numpy as np


fN = "data_new.csv"

trainList = []
valList = []
testList = []

with open(fN,'r') as f:
    csvObj = csv.reader(f)
    for line in csvObj:
        chance = np.random.uniform()
        if chance <= 0.8:
            trainList.append(line)
        elif chance >= 0.8 and chance <= 0.9:
            valList.append(line)
        else:
            testList.append(line)

train = open("train_new.csv",'w',newline='')
val = open("val_new.csv",'w',newline='')
test = open("test_new.csv",'w',newline='')

trainCSV = csv.writer(train)
valCSV = csv.writer(val)
testCSV = csv.writer(test)

trainCSV.writerows(trainList)
valCSV.writerows(valList)
testCSV.writerows(testList)

train.close()
val.close()
test.close()