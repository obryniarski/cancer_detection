from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
import tensorflow as tf
set_random_seed(2)


import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from model import CNN, CNN_experiment
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
data_portion = 4400
batch_size = 32
epochs = 20
names = ['4096/4096/800', '8192/8192/1600']
# val_X, val_y = get_val_data(10000)

data = pd.read_csv('/Users/Oliver/Desktop/datasets/cancer_data/train_labels.csv').sample(data_portion)
path = '/Users/Oliver/Desktop/datasets/cancer_data/train/'


datagen = ImageDataGenerator(rotation_range=45,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    validation_split=0.2,
                                    rescale= 1. / 255.)

train_generator = datagen.flow_from_dataframe(data, path, 'id', 'label', has_ext=False, target_size=(96, 96),
                                            class_mode='binary', subset='training', batch_size=batch_size, shuffle=True)

val_generator = datagen.flow_from_dataframe(data, path, 'id',
                                            'label', has_ext=False, target_size=(96, 96),
                                            class_mode='binary',
                                            subset='validation', batch_size=batch_size)


# ------ Callbacks -------
annealer = LearningRateScheduler(lambda x: 3e-4 * (0.95 ** (x // 3)))
checkpoint = ModelCheckpoint('checkpoint23456.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=3, verbose=1)

# lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-2, steps_per_epoch=data_portion//batch_size, epochs=epochs)
models = [0] * len(names)
histories = [0] * len(names)
for i in range(len(names)):
    models[i] = CNN_experiment(i, input_shape)
    print('--------- Training Model {} ---------'.format(names[i]))
    histories[i] = models[i].fit_generator(train_generator,
                                steps_per_epoch = int(data_portion * 0.8) // batch_size, epochs=epochs,
                                validation_data=val_generator,
                                validation_steps = int(data_portion * 0.2) // batch_size,
                                callbacks=[lr_reducer], verbose=1)




plt.subplot(1, 2, 1)
for history in histories:
    plt.plot(history.history['val_acc'])
    plt.xlim(0, epochs)
    plt.ylim(0.5, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
plt.title('Graph of experiment val_acc')
plt.legend(names)

plt.subplot(1, 2, 2)

for history in histories:
    plt.plot(history.history['loss'])
    plt.xlim(0, epochs)
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
plt.legend(names)
plt.title('Graph of experiment loss')

plt.show()
