#
# @author Gleb Promokhov gleb.promokhov@gmail.com
#

import keras
from keras.layers import GlobalMaxPooling1D, Input, Bidirectional, Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.text import one_hot

import numpy as np

import time
import json
import os

class MultiClassClassifier(object):

    def __init__(self, glove_vectors_path, classes='', training_data='', epochs=10):

        # text preprocessing+embedding
        self.glove_vectors_path = glove_vectors_path
        self.max_most_common_words = 1000 # when tokenize, only keep # most common words
        self.max_sequence_length = 300 # keep only first # words of each training data sentence
        self.max_embeddings_length = 100 # consider only first # gloVe embeddings vector values
        # max_sequence_length >= max_embeddings_length, required?

        # Convolution
        self.filters = 128
        self.kernel_size = 5
        self.pool_size = 35

        # Training
        self.batch_size = 30 # should equal convolution filter size?
        self.epochs = epochs
        self.validation_split = 0.2

        if len(classes) != 0: #meaning do load_from_file() instead

            # check number of classes
            if len(set(classes)) < 2:
                raise ValueError('More than 1 type of label required')

            # find majority class, to use as baseline
            class_counts = {}
            majority_class = ''
            majority_class_count = 0
            for c in classes:
                if c not in class_counts.keys():
                    class_counts[c] = 0
                class_counts[c] += 1
            for c in class_counts:
                if class_counts[c] > majority_class_count:
                    majority_class_count = class_counts[c]
                    majority_class = c

            # set classes data
            self.classes = list(set(classes))
            self.class_counts = class_counts
            self.majority_class = majority_class
            # convert classes into vector
            self.numeric_classes, self.num_to_class = self._make_numeric_classes(classes)
            self.num_classes = len(self.num_to_class)
            # train model
            self.model = self._train(classes, training_data)


    def __repr__(self):
        #return '<MultiClassClassifier: %s rules>' % (len(self._decision_list))
        return '<MultiClassClassifier>'

    def predict_sentence(self, text):
        if not isinstance(text, list):
            test_data = [text]
        guess = self.model.predict(self._vectorize_texts(text))
        guess = np.argmax(guess[0]) # grab the highest likely class
        return self.num_to_class[guess]

    def evaluate(self, test_classes, test_data):

        test_classes, _ = self._make_numeric_classes(test_classes)
        test_classes = keras.utils.to_categorical(test_classes, self.num_classes)

        test_data, _ = self._vectorize_texts(test_data)

        loss_and_acc = self.model.evaluate(test_data, test_classes, batch_size=self.batch_size)

        return loss_and_acc[1]

    def evaluate_by_majority(self, test_data):
        acc = 0
        for t in test_data:
            if self.predict(t) == self.majority_class:
                acc += 1
        return acc/len(test_data)

    def load_from_file(self, config_path, weights_path):
        self.model = model_from_json(open(config_path, encoding='utf-8').read()) #keras.model.model_from_json
        self.model.load_weights(weights_path)
        self.num_classes = self.model.output_shape[1]

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # should load these from model, hardcode for now
        self.num_to_class = ['PERSONAL_CARE', 'MONEY_ISSUE', '(FEAR_OF)_PHYSICAL_PAIN', 'OUTDOOR_ACTIVITY', 'GOING_TO_PLACES', 'ATTENDING_EVENT', 'COMMUNICATION_ISSUE', 'LEGAL_ISSUE']


    def _train(self, classes, training_data):

        # convert classes vector to binary class matrix
        classes = keras.utils.to_categorical(self.numeric_classes, self.num_classes)

        # vectorize training data
        training_data, word_index = self._vectorize_texts(training_data)

        # split data into training/validation data & classes
        training_data, training_classes, validation_data, validation_classes = self._split_data(classes, training_data)

        # load pretrained embeddings {'word':[vector], ...}
        embeddings = self._get_glove_embeddings()

        # embed training data vectors into glove embedding space
        embedded_matrix = self._embed_training_data(embeddings, training_data, word_index)

        # print some info for sanity check
        print('Shape of training data tensor: ', training_data.shape)
        print('Shape of classes tensor: ', classes.shape)

        # build model
        model = self._build_model(embedded_matrix, word_index)

        # fit model
        model.fit(training_data, training_classes,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(validation_data, validation_classes))

        return model

    def save(self, weight_path, config_path):
        self.model.save_weights(weight_path)
        with open(config_path, 'w+') as config:
            config.write(self.model.to_json())

    def _split_data(self, classes, training_data):

        print("CLASSES: ", classes.shape)
        print("training_data: ", training_data.shape)

        # shuffle shuffle!
        shuffle_indices = np.arange(training_data.shape[0])
        np.random.shuffle(shuffle_indices)
        training_data = training_data[shuffle_indices]
        classes = classes[shuffle_indices]

        # split
        split = int(self.validation_split * training_data.shape[0])

        return (training_data[:-split],
               classes[:-split],
               training_data[-split:],
               classes[-split:])

    def _make_numeric_classes(self, classes):
        numeric_classes = []
        num_to_class = list(set(classes))
        for c in classes:
            numeric_classes.append(num_to_class.index(c))
        return numeric_classes, num_to_class

    def _build_model(self, embedded_matrix, word_index):

        model = Sequential()

        print("WORDINDEX: %s" % len(word_index))
        print("max_embeddings_length: %s" % self.max_embeddings_length)
        print("input_length: %s" % self.max_sequence_length)
        num_words = min(self.max_most_common_words, len(word_index))

        # define model using Keras functional API
        # look harder, but it makes it wayyyy easiser to arrange layres then with Sequential()

        sequence_input = Input(shape=(self.max_sequence_length, ), dtype='int32')

        # using pre-trained embeddings
        embedding_layer = Embedding(num_words,
                                    self.max_embeddings_length,
                                    weights=[embedded_matrix],
                                    input_length=self.max_sequence_length,
                                    trainable=False)

        # train embeddings outselves
        # embedding_layer = Embedding(num_words, self.max_embeddings_length, input_length=self.max_sequence_length)

        embedded_sequences = embedding_layer(sequence_input)

        m = Conv1D(self.filters, self.kernel_size, activation='relu')(embedded_sequences)
        m = MaxPooling1D(self.kernel_size)(m)
        m = Conv1D(self.filters, self.kernel_size, activation='relu')(m)
        m = MaxPooling1D(self.kernel_size)(m)

        m = GlobalMaxPooling1D()(m)

        m = Dense(self.filters, activation='relu')(m)
        predictions = Dense(self.num_classes, activation='softmax')(m)

        model = Model(sequence_input, predictions)

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    def _get_glove_embeddings(self):

        embeddings_index = {}
        with open(self.glove_vectors_path) as embeds_file:
            for line in embeds_file:
                line = line.split()
                word = line[0]
                coefs = np.asarray(line[1:], dtype='float32')
                embeddings_index[word] = coefs

        return embeddings_index

    def _embed_training_data(self, embeddings, training_data, word_index):

        num_words = min(self.max_most_common_words, len(word_index))
        matrix = np.zeros((num_words, self.max_embeddings_length))
        for word, i in word_index.items():
            if i >= self.max_most_common_words:
                continue
            vector = embeddings.get(word)
            if vector is not None:
                matrix[i] = vector #otherwise all zeros

        return matrix


    def _vectorize_texts(self, texts):

        tokenizer = Tokenizer(num_words=self.max_most_common_words) # only consider this many most commmon words
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, self.max_sequence_length)

        return data, tokenizer.word_index
