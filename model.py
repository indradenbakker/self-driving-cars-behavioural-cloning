### Importing packages
import os
import pandas as pd
import numpy as np
from scipy import signal
import cv2
import math
import tensorflow as tf
tf.python.control_flow_ops = tf



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

