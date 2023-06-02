import csv
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of the training set.
    tr_y = None  # Y (label) of the training set.
    ts_x = None  # X (data) of the test set.
    ts_y = None  # Y (label) of the test set.
    label_encoder = None  # Label encoder object

    def __init__(self, trainfile, testfile):

        if not os.path.isfile(trainfile) or not os.path.isfile(testfile):
            raise FileNotFoundError("File '{}' not found.".format(trainfile))
        
        tr_x = []  # List to store training images
        tr_y = []  # List to store training labels
        ts_x = []  # List to store test images
        ts_y = []  # List to store test labels

        with open(trainfile, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if 'train' in row[0]:
                    img_path, label = row[0], row[1]
                    img = np.asarray(Image.open(os.path.join("dataset", img_path)))
                    tr_x.append(img)
                    tr_y.append(label)


        with open(testfile, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if 'test' in row[0]:
                    img_path, label = row[0], row[1]
                    img = np.asarray(Image.open(os.path.join("dataset", img_path)))
                    ts_x.append(img)
                    ts_y.append(label)


        # Convert the lists to numpy arrays
        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y)
        self.ts_x = np.array(ts_x)
        self.ts_y = np.array(ts_y)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.concatenate((self.tr_y, self.ts_y), axis=0))

        self.tr_y = self.label_encoder.transform(self.tr_y)
        self.ts_y = self.label_encoder.transform(self.ts_y)

        print(len(self.tr_x))
        print(len(self.tr_y))
        print(len(self.ts_x))
        print(len(self.ts_y))

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.ts_x, self.ts_y]
    
    def getClasses(self):
        return np.unique(np.concatenate((self.tr_y, self.ts_y), axis=0)).tolist()

# # Example usage:
# csv_file = 'data/cifar100_nl.csv'  # Path to your CIFAR-100 NL test CSV file
# loader = C100Dataset("data/cifar100_nl.csv", 'data/cifar100_nl_test.csv')
# [data_nl_tr_x, data_nl_tr_y, data_nl_val_x, data_nl_val_y] = loader.getDataset()