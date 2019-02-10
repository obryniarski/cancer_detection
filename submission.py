from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


datagen = ImageDataGenerator(
            rescale=1. / 255.)

test_generator = datagen.flow_from_directory('/Users/Oliver/Desktop/datasets/cancer_data', target_size=(96, 96),
                                        class_mode=None, shuffle=False, classes=['test'],
                                        batch_size=1)


model = load_model('checkpoint.h5', custom_objects={'auroc': auroc})
# checkpoint_model = CNN()
# checkpoint_model.load_weights('checkpoint.h5')

model = model

# files = [a[2] for a in os.walk('data/test') if a[0] == 'data/test'][0]
files = test_generator.filenames
print(files[0][5:-4])
y = model.predict_generator(test_generator, len(files), verbose=1)

submission = pd.DataFrame({'id': [file[5:-4] for file in files], 'label': y.reshape(y.shape[0])})
print(submission.head())

submission.to_csv('submission.csv', index=False)
