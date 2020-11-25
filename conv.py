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
import tensorflow.keras.layers as Layers
import tensorflow.keras.losses as Losses
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#%% Define funs
# List all arrays
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
    batch_x = tf.cast(batch_x, dtype=tf.float32)
    yield (batch_x, batch_x)

#%% Train-test split

INPUT_SIZE = (256, 512, 1)

train, test = train_test_split(arrays, test_size=.1, random_state=SEED)

#%% Define model

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      Layers.Input(INPUT_SIZE),
      
      Layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
      Layers.MaxPool2D(),
      Layers.BatchNormalization(),
      
      Layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
      Layers.MaxPool2D(),
      Layers.BatchNormalization(),
      
      Layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
      Layers.MaxPool2D(),
      Layers.BatchNormalization(),
      
      Layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
      Layers.MaxPool2D(),
      Layers.BatchNormalization(),
      
      Layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
      Layers.MaxPool2D(),
      Layers.BatchNormalization(),

      Layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
      Layers.MaxPool2D(),
      Layers.BatchNormalization(),
      
      Layers.Flatten(),
      Layers.Dense(32),
      Layers.Dropout(.2),
      Layers.Dense(latent_dim, activation='linear'),
    ])
    self.decoder = tf.keras.Sequential([
      Layers.Dense(32, activation='relu'),
      Layers.Dropout(.2),
      Layers.Dense(2048, activation='relu'),
      Layers.Reshape((4, 8, 64)),
      Layers.BatchNormalization(),
      Layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
      Layers.UpSampling2D(2),
      
      Layers.BatchNormalization(),
      Layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
      Layers.UpSampling2D(2),

      Layers.BatchNormalization(),
      Layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
      Layers.UpSampling2D(2),

      Layers.BatchNormalization(),
      Layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
      Layers.UpSampling2D(2),

      Layers.BatchNormalization(),
      Layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
      Layers.UpSampling2D(4),

      Layers.Conv2D(1, (3, 3), padding='same', activation='relu'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim=CODE_SIZE)

#%% Compile
autoencoder.compile(optimizer='adam',
                loss=Losses.MeanSquaredError())

#%% Train
autoencoder.fit(image_generator(),
                epochs=NUM_EPOCHS, 
                validation_data=image_generator())
                

prediction = autoencoder.encoder.predict(test)