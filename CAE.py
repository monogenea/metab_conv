#%% Explore the npy arrays

SEED = 123
TARGET_SIZE = (512, 256)
# Model pars
CODE_SIZE = 2
NUM_EPOCHS = 10
BATCH_SIZE = 32
DIR = '/Users/franciscolima/Documents/Projects/metab_conv/'
IMG_PATH = DIR + 'arrays/'

import os, re, glob, cv2
os.chdir(DIR)
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

#%% Define funs

arrays = glob.glob(IMG_PATH + '*.npy')

def get_input(fpath):
    arr = np.load(fpath)
    # Resize imgs
    blurr = cv2.blur(arr, ksize=(5, 5))
    arr_resized = cv2.resize(blurr, dsize=TARGET_SIZE)
    arr_resized = arr_resized[:,:,np.newaxis]
    return arr_resized

def image_generator(files=arrays, batch_size=BATCH_SIZE):
    # Take random batch
    batch = np.random.choice(a=files, size=batch_size)
    batch_x = [get_input(f) for f in batch] #np.stack([get_input(f) for f in batch], axis=3)
    return batch_x

# im_ds = tf.keras.preprocessing.image.image_dataset_from_directory('arrays/',
#     label_mode=None,
#     color_mode='grayscale', batch_size=32, seed=SEED, validation_split=0.1)


# im_dg = tf.keras.preprocessing.image.ImageDataGenerator(

# tf.keras.preprocessing.image.NumpyArrayIterator(
#     x, y, im_dg, batch_size=32, shuffle=False, sample_weight=None,
#     seed=SEED, data_format=None, save_to_dir=None, save_prefix='',
#     save_format='png', subset=None, dtype=None)

# class Autoencoder(Model):
#   def __init__(self, latent_dim):
#     super(Autoencoder, self).__init__()
#     self.latent_dim = latent_dim   
#     self.encoder = tf.keras.Sequential([
#       layers.Flatten(),
#       layers.Dense(latent_dim, activation='relu'),
#     ])
#     self.decoder = tf.keras.Sequential([
#       layers.Dense(784, activation='sigmoid'),
#       layers.Reshape((28, 28))
#     ])

#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded

# autoencoder = Autoencoder(CODE_SIZE)


# autoencoder.compile(optimizer='adam',
#                 loss=losses.MeanSquaredError())

# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))