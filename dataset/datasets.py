import csv
import numpy as np
from PIL import Image

class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of the training set.
    tr_y = None  # Y (label) of the training set.
    ts_x = None  # X (data) of the test set.
    ts_y = None  # Y (label) of the test set.

    def __init__(self, train_filename, test_filename):
        # Read the CSV for the training set
        tr_x = []  # List to store training images
        tr_y = []  # List to store training labels

        with open(train_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                img_path, label = row[0], row[1]
                # Perform any necessary preprocessing on the image or label
                tr_x.append(img_path)
                tr_y.append(label)

        # Read the CSV for the test set
        ts_x = []  # List to store test images
        ts_y = []  # List to store test labels

        with open(test_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                img_path, label = row[0], row[1]
                # Perform any necessary preprocessing on the image or label
                ts_x.append(img_path)
                ts_y.append(label)

        # Convert the lists to numpy arrays if desired
        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y)
        self.ts_x = np.array(ts_x)
        self.ts_y = np.array(ts_y)

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.ts_x, self.ts_y]