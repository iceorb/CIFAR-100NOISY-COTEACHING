from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision.io import read_image

class C100Dataset(Dataset):

    tr_x =[]
    tr_y =[]
    ts_x =[]
    ts_y =[]
    classes = {}

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, train = False, test = False):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            root_dir (string): Root directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_info = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        pd.read_csv(csv_file)

        # if train is true store it in tr_x and tr_y for all the images and labels in csv file
        if train is True:
            self.tr_x = self.data_info.iloc[:, 0]
            self.tr_y = self.data_info.iloc[:, 1]

        if test is True:
            self.ts_x = self.data_info.iloc[:, 0]
            self.ts_y = self.data_info.iloc[:, 1]

        # Create classes / build more classes
        self._createClassDict()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data_info.iloc[index, 0])
        image = read_image(img_path)
        label = self.data_info.iloc[index, 1]


        if self.transform:
            image = self.transform(image)

        label = self.classes[label]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def _createClassDict(self):
        unique_labels = self.data_info.iloc[:, 1].unique()
        for i, label in enumerate(unique_labels):
            self.classes[label] = i
    
    def getClasses(self):
        return self.classes
    
    def getDataset(self):
        return [self.tr_x, self.tr_y, self.ts_x, self.ts_y]
    
##  Example usage:
# import datasets
# from torchvision.transforms import ToTensor as toTensor
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# from torchvision.transforms import ToPILImage
# import matplotlib.pyplot as plt
# import torch
# from PIL import Image

# transform = transforms.Compose([
#     transforms.Resize((32,32)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
#     # Using previous information, skipping manual calculation
# ])

# target_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# trainDataset = datasets.C100Dataset('dataset/data/cifar100_nl.csv', 
#                                     root_dir="dataset",
#                                     train=True,
#                                     transform= transform)

# testDataset = datasets.C100Dataset('dataset/data/cifar100_nl_test.csv',
#                                     root_dir="dataset",
#                                     test=True,
#                                     transform = transform,
#                                     target_transform=target_transform)

# trainDataset, valDataset = random_split(trainDataset, [.8, .2])

# trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
# valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
# testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)
