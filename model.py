### Importing packages
import os
import pandas as pd
import numpy as np
from scipy import signal
import cv2
import math
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras import initializations

from pathlib import Path
import json

img_cols = 64
img_rows = 64
img_channels = 3

def save_model(fileModelJSON, fileWeights):
    # If the file already exists, first delete the file before saving the new model 
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)

def preprocess_image(name):
    # Preprocessing image
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def create_model():
    input_shape = (img_rows, img_cols, img_channels)
    filter_size = 3

    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=input_shape))
    model.add(Convolution2D(3, 1, 1, border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(64, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(16, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1, init='he_normal'))
    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=32, help='Batch size.')
    ap.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
    ap.add_argument('--overwrite', type=bool, default=False, help='Overwrite old model.')
    args = vars(ap.parse_args())

    nb_epoch = args['epoch']
    batch_size = args['batch']
    overwrite = args['overwrite']

    np.random.seed(2016)

    load_csv()

    define_model()

    train_model()

