
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics.scorer import roc_auc_scorer
from sklearn.metrics import accuracy_score
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

class LogisticRegressionBinaryClassifier(object):

    def __init__(self, classes, training_data):

        if len(set(classes)) != 2:
            raise ValueError("Can only have 2 classes")

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

        self._majority_class = majority_class

        self._classes = list(set(classes))
        self._true_false_classes = self._convert_classes(classes)

        self._vectorizer = TfidfVectorizer(use_idf=False, norm='l1')

        self._model = self._train(training_data)

    def predict(self, text):
        if self._model.predict(text) == True:
            return self._classes[0]
        return self._classes[1]

    def evaluate(self, test_classes, test_data):
        test_data = self._vectorizer.transform(test_data)
        predicts = []
        for t in test_data:
            predicts.append(self.predict(t))
        return accuracy_score(predicts, test_classes)

    def evaluate_by_majority(self, test_classes, test_data):
        acc = 0
        test_data = self._vectorizer.transform(test_data)
        for t in test_data:
            if self.predict(t) == self._majority_class:
                acc += 1
        return acc/len(test_classes)

    def _train(self, training_data):
        classes = np.array(self._true_false_classes)
        features = self._vectorizer.fit_transform(training_data)
        lr = LogisticRegressionCV(Cs=10, class_weight='balanced',
                              scoring='accuracy', solver='sag',
                              tol=0.001, max_iter=500,
                              random_state=0)
        lr.fit(features, classes)
        return lr

    def _convert_classes(self, classes):
        cl = list(set(classes))
        output = []
        for c in classes:
            output.append(cl[0] == c)
        return output

    def __repr__(self):
        return '<LogisticRegressionBinaryClassifier>'
