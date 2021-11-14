import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file("kanye_verses.txt")