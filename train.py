from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from model import CNN
import keras.backend as K
from keras.callbacks import Callback


class LRFinder(Callback):

    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr-self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()

    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()

input_shape = (96, 96, 3)
data_portion = 220000
batch_size = 32
epochs = 6
# val_X, val_y = get_val_data(10000)
try:
    data = pd.read_csv('/Users/OliverBryniarski 1/Desktop/datasets/cancer_data/train_labels.csv').sample(data_portion)
    path = '/Users/OliverBryniarski 1/Desktop/datasets/cancer_data/train'
except:
    data = pd.read_csv('/floyd/input/data/train_labels.csv').sample(data_portion)
    path = '/floyd/input/data/train'

datagen = ImageDataGenerator(rotation_range=45,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    validation_split=0.2,
                                    rescale= 1. / 255.)


train_generator = datagen.flow_from_dataframe(data, path, 'id',
                                            'label', False, (96,96),
                                            class_mode='binary',
                                            subset='training', batch_size=batch_size, shuffle=True)

val_generator = datagen.flow_from_dataframe(data, path, 'id',
                                            'label', False, (96,96),
                                            class_mode='binary',
                                            subset='validation', batch_size=batch_size)


# ------ Callbacks -------
annealer = LearningRateScheduler(lambda x: 3e-4 * (0.95 ** (x // 3)))
checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=3, verbose=1)


# lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-2, steps_per_epoch=data_portion//batch_size, epochs=epochs)

model = CNN(input_shape)
history = model.fit_generator(train_generator,
                            steps_per_epoch = int(data_portion * 0.8) // batch_size, epochs=epochs,
                            validation_data=val_generator,
                            validation_steps = int(data_portion * 0.2) // batch_size,
                            callbacks=[lr_reducer, checkpoint], verbose=1)

print(model.summary())
model.save('my_model.h5')
# lr_finder.plot_loss()


plt.subplot(1,2,1)
plt.plot(history.history['val_acc'])
plt.xlim(0, epochs)
plt.ylim(0.5, 1)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title(max(history.history['val_acc']))

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.xlim(0, epochs)
plt.ylim(0, 1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(min(history.history['loss']))
plt.show()
