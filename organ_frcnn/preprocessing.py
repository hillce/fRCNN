import csv

import numpy as np

# TODO: Split based on seeding

def train_test_split_csv(
    csv_fN: str,
    train_prop=0.8,
    test_prop=0.1,
    seed = 42) -> None:
    """Function for randomly splitting the lines in a csv into train, val and test csvs.
    If the sum of train_prop and test_prop != 1.0, then a validation set will be created.

    Args:
        csv_fN (str): path to data csv file
        train_prop (float, optional): train proportion. Defaults to 0.8.
        test_prop (float, optional): test proportion. Defaults to 0.1.
        seed (int, optional): the seed to make splitting reproducible
    """

    trainList = []
    valList = []
    testList = []

    with open(csv_fN,'r') as f:
        csvObj = csv.reader(f)
        for line in csvObj: 
            chance = np.random.uniform()
            if chance <= train_prop:
                trainList.append(line)
            elif chance >= train_prop and chance <= (1-test_prop):
                valList.append(line)
            else:
                testList.append(line)

    with open(f"{csv_fN[:-4]}_train.csv",'w',newline='') as train:
        trainCSV = csv.writer(train)
        trainCSV.writerows(trainList)

    if (train_prop + test_prop) != 1:
        with open(f"{csv_fN[:-4]}_val.csv",'w',newline='') as val:
            valCSV = csv.writer(val)
            valCSV.writerows(valList)

    with open(f"{csv_fN[:-4]}_test.csv",'w',newline='') as test:
        testCSV = csv.writer(test)
        testCSV.writerows(testList)
