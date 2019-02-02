
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from keras.utils import Sequence
import math
from sklearn.model_selection import train_test_split

def get_val_data(num=None):
    train_directory_path = 'data/train/*.tif'
    train_labels = pd.read_csv('data/train_labels.csv')
    images = glob.glob(train_directory_path)
    if not num:
        num = len(images)
        print(num)

    width = 96
    height = 96
    channels = 3
    X = np.zeros((num, width, height, channels), np.int16)
    y = np.zeros((num, 1))
    print(X.shape)

    for i in range(num):
        im_id = images[i][11:-4]
        label = train_labels.loc[train_labels['id'] == im_id]['label'].iloc[0]

        X[i] = plt.imread(images[i])
        y[i] = np.array(label)
        # plt.imshow(im)
        # plt.show()

    _, test_X, _, test_y = train_test_split(X, y, test_size=0.99)
    print(test_X.shape)
    return test_X / 255, test_y


def get_data_generator():
    train_directory_path = 'data/train/*.tif'
    train_labels = pd.read_csv('data/train_labels.csv')
    images = glob.glob(train_directory_path)
    num = len(images)

    width = 96
    height = 96
    channels = 3

    for i in range(num):
        im_id = images[i][11:-4]
        label = train_labels.loc[train_labels['id'] == im_id]['label'].iloc[0]

        im = plt.imread(images[i])
        # plt.imshow(im)
        # plt.show()
        yield im, label


train_directory_path = 'data/train/*.tif'
labels = pd.read_csv('data/train_labels.csv')
images = glob.glob(train_directory_path)



class DataSequence(Sequence):

    def __init__(self, batch_size=32, augment=False):

        self.x, self.y = images, labels
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # batch containing file names of images
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]

        if len(batch_x) < self.batch_size:
            batch_size = len(batch_x)
        else:
            batch_size = self.batch_size

        # contains ids of the batch of images (for use in getting batch of y's)
        batch_x_ids = [batch_x[i][11:-4] for i in range(batch_size)]
        # print([batch_x_ids[i] for i in range(batch_size)])

        # contains batch of labels corresponding to the batch of x's
        batch_y = [self.y.loc[self.y['id'] == batch_x_ids[i]]['label'].iloc[0] for i in range(batch_size)]
        batch_y = np.array(batch_y)

        # read file names to make batch_x an array containing image arrays
        batch_x = np.array([plt.imread(file_name) for file_name in batch_x])
        batch_y = batch_y.reshape((batch_y.shape[0], 1))

        if self.augment:
            batch_x = self.augment(batch_x)

        return batch_x / 255, batch_y

    # def on_epoch_end(self):
    #     self.indexes =

    def augment(batch): # augment a batch and return the new batch, same order/size
        return None


# test = DataSequence(64)
#
# print(test[800][0].shape, test[800][1], test[800][1].shape)


# class ValSequence(Sequence):
#
#     def __init__(self, batch_size=32, augment=False):
#
#         self.x, self.y = test_images, test_labels
#         self.batch_size = batch_size
#         self.augment = augment
#
#     def __len__(self):
#         return math.ceil(len(self.x) / self.batch_size)
#
#     def __getitem__(self, idx):
#         # batch containing file names of images
#         batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
#         if len(batch_x) < self.batch_size:
#             batch_size = len(batch_x)
#         else:
#             batch_size = self.batch_size
#         # contains ids of the batch of images (for use in getting batch of y's)
#         batch_x_ids = [batch_x[i][11:-4] for i in range(batch_size)]
#         print([batch_x_ids[i] for i in range(batch_size)])
#
#         # contains batch of labels corresponding to the batch of x's
#         batch_y = [self.y.loc[self.y['id'] == batch_x_ids[i]]['label'].iloc[0] for i in range(batch_size)]
#         batch_y = np.array(batch_y)
#
#         # read file names to make batch_x an array containing image arrays
#         batch_x = np.array([plt.imread(file_name) for file_name in batch_x])
#         batch_y = batch_y.reshape((batch_y.shape[0], 1))
#
#         if self.augment:
#             batch_x = self.augment(batch_x)
#
#         return batch_x, batch_y
