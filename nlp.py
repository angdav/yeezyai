import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
import time

from tensorflow.python.util import lazy_loader

# creates array of each line as a separate sentence
with open('kanye_verses.txt') as f:
    corpus = f.readlines()

# vocab_size = 1000  # lower vocab size for more accurate meaning
# embedding_dim = 16  # dimension of embedding
# max_length = 16  # each bar shouldn't be that long
# trunc_type = "post"  # we want extra words to come at the end
# padding_type = "post"
# oov_token = "<OOV>"
# training_size = 20000  # may increase in the future

# # creates dictionary of words assigned to tokens (enumerate each word for data manipulation)
# # oov_token accounts for unfamiliar words
# tokenizer = Tokenizer(num_words=10000, oov_token=oov_token)
# tokenizer.fit_on_texts(lines)
# word_index = tokenizer.word_index

# # create uniformity in length of sentences
# train_sequences = tokenizer.texts_to_sequences(lines)
# train_padded = pad_sequences(
#     train_sequences, padding=padding_type, maxlen=max_length)

# # encode validation in addition to training models
# validation_sequences = tokenizer.texts_to_sequences(lines)
# validation_padded = pad_sequences(
#     validation_sequences, padding=padding_type, maxlen=max_length)

# label_tokenizer = Tokenizer()
# label_tokenizer.fit_on_texts(labels)

# training_label_sequence = np.array(
#     label_tokenizer.texts_to_sequences(train_labels))
# validation_label_sequence = np.array(
#     label_tokenizer.texts_to_sequences(validation_labels))

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(
#         vocab_size, 64),
#     tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])
# model.summary()

# num_epochs = 30
# history = model.fit(train_padded, training_label_sequence, epochs=num_epochs,
#                     validation_data=(validation_padded, validation_label_sequence), verbose=2)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # need total_words count for later

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# according to data, max length allowed for each line
max_sequence_len = max([len(x) for x in input_sequences])

# pad sequences with 0s at the beginning, to have training data on left of input, and label to the right
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))

# Assign the input and labels according to padding
# xs represent input to the function
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]

# ys represent output to the function (basic RNN concept, most recent data is most important)
# this is where LSTM comes intp play
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(
    total_words, 64, input_length=max_sequence_len - 1))
# Uses bound of 20 words to affect generation of singular word
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)))
# Dense layer neural network
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
# Optimization algorithm using stochastic gradient descent
adam = tf.keras.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=adam, metrics=['accuracy'])
# Lowered epochs drastically to test plausibility at first, seems not going anywhere
model.fit(xs, ys, epochs=10, verbose=1)

seed_text = "She said"
next_words = 10  # generate 10 words after initial seed

# Generates each word in 10 after initial seed
# Pads each line with empty words to bring to same size
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

f = open("nlp-output.txt", "a")
f.write(seed_text + "\n")
f.close()
