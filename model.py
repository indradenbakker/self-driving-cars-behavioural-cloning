### Importing packages
import os
import argparse
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
from keras.utils.visualize_util import plot

from pathlib import Path
import json

img_cols = 64
img_rows = 64
img_channels = 3

def save_model(path_model_json, path_model_weights):
    # If the file already exists, first delete the file before saving the new model 
    if Path(path_model_json).is_file():
        os.remove(path_model_json)
    json_string = model.to_json()
    with open(path_model_json,'w' ) as f:
        json.dump(json_string, f)
    if Path(path_model_weights).is_file():
        os.remove(path_model_weights)
    model.save_weights(path_model_weights)

def augment_image(image):
    # Change image type for augmentation
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    # Add random augmentation
    random_bright = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return(image)

def preprocess_image(data):
    # Randomly pick left, center or right camera image
    rand = np.random.randint(3)
    if (rand == 0):
        path_file = data['left'][0].strip()
        shift_ang = .25
    if (rand == 1):
        path_file = data['center'][0].strip()
        shift_ang = 0.
    if (rand == 2):
        path_file = data['right'][0].strip()
        shift_ang = -.25
    y = data['steering'][0] + shift_ang

    # Read image
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop image
    shape = image.shape
    image = image[math.floor(shape[0]/4):shape[0]-20, 0:shape[1]]

    # Resize image
    image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_AREA) 

    # Augment image
    image = augment_image(image)
    image = np.array(image)

    if np.random.choice([True, False]):
        image = cv2.flip(image,1)
        y = -y
    
    return(image, y)

def batchgen(data, batch_size):
    # Create empty numpy arrays
    batch_images = np.zeros((batch_size, img_rows, img_cols, img_channels))
    batch_steering = np.zeros(batch_size)

    small_steering_threshold = 0.8

    # Custom batch generator 
    while 1:
        for n in range(batch_size):
            i = np.random.randint(len(data))
            data_sub = data.iloc[[i]].reset_index()
            # Only keep training data with small steering angles with probablitiy 
            keep = False
            while keep == False:
                x, y = preprocess_image(data_sub)
                pr_unif = np.random
                if abs(y) < .01:
                    next;
                if abs(y) < .10:
                    small_steering_rand = np.random.uniform()
                    if small_steering_rand > small_steering_threshold:
                        keep = True
                else:
                    keep = True

            batch_images[n] = x
            batch_steering[n] = y
        yield batch_images, batch_steering

def train_model(model, data, batch_size, epochs, overwrite, val_size):

    # Default best epoch and validation loss
    n_best = 0
    val_best = 1e5

    # Loop over epochs for training and saving results
    for n in range(nb_epoch):
        train_generator = batchgen(data, batch_size)
        val_generator = batchgen(data, batch_size)
        
        history = model.fit_generator(train_generator, samples_per_epoch=24960, nb_epoch=1, validation_data=val_generator, nb_val_samples=val_size)
        
        # Save model and weights to file
        path_model_json = 'model_' + str(n) + '.json'
        path_model_weights = 'model_' + str(n) + '.h5'        
        save_model(path_model_json, path_model_weights)
        
        # Add validation loss to history
        val_loss = history.history['val_loss'][0]

        # Keep track of lowest validation loss
        if val_loss < val_best:
            i_best = n 
            val_best = val_loss
            path_model_best = 'model_best.json'
            path_weights_best = 'model_best.h5'
            save_model(path_model_best, path_weights_best)

    # Print results
    print('Best model found at iteration # ' + str(i_best))
    print('Best Validation score : ' + str(np.round(val_best,4)))

    return(history)


def define_model():
    # Create model
    input_shape = (img_rows, img_cols, img_channels)
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

    # Set parameters
    # Learning rate of 0.0001 emperically showed best result: low enough to get the network to converge and learn to drive on the track
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # Compile model
    model.compile(optimizer=adam, loss='mse')

    # Output summary of the model
    model.summary()

    # Visualise model and export to file
    plot(model, to_file='model.png')

    return(model)

def load_csv(csv_path='driving_log_ud.csv'):
    # By default use path of CSV data generated by Udacity
    data = pd.read_csv(csv_path, header=None, skiprows=[0], names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    return(data)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=32, help='Batch size.')
    ap.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
    ap.add_argument('--overwrite', type=bool, default=False, help='Overwrite old model.')
    ap.add_argument('--val', type=float, default=0.2, help='Percentage of data for validation')
    args = vars(ap.parse_args())

    nb_epoch = args['epoch']
    batch_size = args['batch']
    overwrite = args['overwrite']
    val_size = args['val']

    np.random.seed(2016)

    data = load_csv()

    model = define_model()
    
    history = train_model(model, data, batch_size, nb_epoch, overwrite, val_size)

