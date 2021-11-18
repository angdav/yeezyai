import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
import os
import time

# creates array of each line as a separate sentence
with open('kanye_verses.txt') as f:
    lines = f.readlines()

# creates dictionary of words assigned to tokens (enumerate each word for data manipulation)
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(lines)
word_index = tokenizer.word_index
print(word_index)
