from __future__ import print_function
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Dense, Activation, LSTM, Bidirectional, Embedding

with open('kanye_verses.txt') as f:
    corpus = f.readlines()

# Dictionary linking each word to its count in the dataset
frequencies = {}
# Set of words indicating words that appear less than the MIN_FREQUENCY
uncommon_words = set()

MIN_FREQUENCY = 10
MIN_SEQ = 5  # Minimum length of bars in each verse
BATCH_SIZE = 32

words = []
for line in corpus:
    words += line.split()

unsorted_words = words  # Store unsorted words for seed creation later

for word in words:
    frequencies[word] = frequencies.get(word, 0) + 1

uncommon_words = set([key for key in frequencies.keys()
                      if frequencies[key] < MIN_FREQUENCY])
words = sorted(set([key for key in frequencies.keys()
                    if frequencies[key] >= MIN_FREQUENCY]))

num_words = len(words)
# Dictionary of words to index (count)
word_indices = dict((w, i) for i, w in enumerate(words))
# Reversed dictionary of index (count) to words
indices_word = dict((i, w) for i, w in enumerate(words))

valid_sequences = []
words_at_end_of_sequences = []

for i in range(len(unsorted_words) - MIN_SEQ):
    end_slice = i + MIN_SEQ + 1
    # If chosen sequence contains no uncommon words, then add it to list of valid sequences
    if len(set(unsorted_words[i:end_slice]).intersection(uncommon_words)) == 0:
        valid_sequences.append(unsorted_words[i: i + MIN_SEQ])
        words_at_end_of_sequences.append(unsorted_words[i + MIN_SEQ])

X_train, X_test, y_train, y_test = train_test_split(
    valid_sequences, words_at_end_of_sequences, test_size=0.02, random_state=42)


# Data generator for fit and evaluate
# Fits words to indexes, much like nlp.py
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, MIN_SEQ), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


output_file = open("nlp2-output.txt", "a")


def on_epoch_end(epoch, logs):
    # Prints the text generated at the end of each epoch
    output_file.write("Epoch: " + str(epoch + 1) + "\n")
    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(X_train+X_test))
    seed = (X_train+X_test)[seed_index]
    # Run a wide spectrum of sampling temperature (randomness)
    for sampling_temperature in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sentence = seed
        output_file.write("Sampling Temperature: " +
                          str(sampling_temperature) + "\n")
        output_file.write(' '.join(sentence))
        for i in range(30):  # Word count of output after seed
            x_pred = np.zeros((1, MIN_SEQ))
            for t, word in enumerate(sentence):
                x_pred[0, t] = word_indices[word]
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, sampling_temperature)
            next_word = indices_word[next_index]
            sentence = sentence[1:]
            sentence.append(next_word)
            output_file.write(" "+next_word)
        output_file.write('\n')
    output_file.write('\n')
    output_file.flush()


# Create the model, much in style of initial NLP approach in nlp.py
model = Sequential()
# Embed words close in meaning together (multi-dimensional transformation)
# Input dimension is number of words, output dimension is 1024 (less)
model.add(Embedding(input_dim=len(words), output_dim=1024))
# Uses bound of 128 words to affect generation of singular word (64 on each side)
model.add(Bidirectional(LSTM(128)))
model.add(Dense(len(words)))
# Decides if a reached neuron should be "activated" (used) in result based on bias and sum
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam", metrics=['accuracy'])

# Print the generation as it runs, each time an epoch is complete
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
callbacks_list = [print_callback]

examples_file = open('nlp2-output.txt', "a")
model.fit(generator(X_train, y_train, BATCH_SIZE),
          steps_per_epoch=int(len(valid_sequences)/BATCH_SIZE) + 1,
          epochs=50,
          callbacks=callbacks_list,
          validation_data=generator(X_test, y_train, BATCH_SIZE),
          validation_steps=int(len(y_train)/BATCH_SIZE) + 1)
