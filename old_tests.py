#
#   Tests for classifiers/
#
#   @author Gleb Promokhov gleb.promohov@gmail.com
#


from classifiers.MultiClassClassifier import MultiClassClassifier
from classifiers.LogisticRegressionBinaryClassifier import LogisticRegressionBinaryClassifier

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.metrics import ConfusionMatrix
from collections import Counter
import re
import string
import random
import json
import time
from sklearn.model_selection import StratifiedKFold
import numpy as np


def read_corpus(fname):
    result = []
    with open(fname) as f:
        for line in f:
            line = line.strip().split("\t")
            result.append({
                "id": line[0],
                "sentence": line[1],
                "genre": line[4],
                "polarity": line[2],
                "topic": line[3]
            })
    return result

def cross_validate_majority(model_maker, classes, texts, num_splits=10):

    kfold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=13)
    accuracy_scores = 0

    classes = np.asarray(classes)
    texts = np.asarray(texts)

    for train, test in kfold.split(texts, classes):
        model = model_maker(classes[train], texts[train])
        accuracy_scores += model.evaluate_by_majority(classes[test], texts[test])

    return accuracy_scores/num_splits

def cross_validate(model_maker, classes, texts, num_splits=10):

    kfold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=13)
    accuracy_scores = 0

    classes = np.asarray(classes)
    texts = np.asarray(texts)

    max_accuracy_score = 0
    max_model = ''

    for train, test in kfold.split(texts, classes):
        model = model_maker(classes[train], texts[train])
        accuracy = model.evaluate(classes[test], texts[test])
        if accuracy > max_accuracy_score:
            max_accuracy_score = accuracy
            max_model = model

    t = time.strftime("%d-%m-%Y")
    max_model.save('models/mcc_model_%s.trn' % (t), 'models/mcc_config_%s.json' % (t))

    return accuracy_scores/num_splits

def make_lrbc(classes, texts):
    return LogisticRegressionBinaryClassifier(classes, texts)



def demo(model):

    corpus = read_corpus('data/training_data.txt')
    classes = [l['genre'] for l in corpus][:200]
    texts = [l['sentence'] for l in corpus][:200]

    num_folds = 10
    accuracy_majority = cross_validate_majority(make_lrbc, classes, texts, num_folds)
    accuracy = cross_validate(make_lrbc, classes, texts, num_folds)

def test0():
    corpus = read_corpus('data/training_data.txt')
    classes = [l['genre'] for l in corpus]
    texts = [l['sentence'] for l in corpus]

    num_folds = 10
    accuracy_majority = cross_validate_majority(make_lrbc, classes, texts, num_folds)
    accuracy = cross_validate(make_lrbc, classes, texts, num_folds)

    print("Task 1, logistic regression, num of cross-validation folds: {}".format(num_folds))
    print("ACCURACY_BY_MAJORITY: ~{}%".format(round(accuracy_majority*100, 2)))
    print("ACCURACY: ~{}%".format(round(accuracy*100, 2)))

def test1(save=False):

    corpus = read_corpus('data/training_data.txt')
    classes = [l['genre'] for l in corpus]
    texts = [l['sentence'] for l in corpus]

    split = len(classes)//40
    train_classes = classes[split:]
    train_texts = texts[split:]
    test_classes = classes[:split]
    test_texts = texts[:split]

    model = make_mcc(train_classes, train_texts)
    accuracy = model.evaluate(test_classes, test_texts)
#
    if save:
        t = time.strftime("%d-%m-%Y")
        model.save('models/mcc_model_%s.trn' % (t), 'models/mcc_config_%s.json' % (t))

    print("Task 1, neural net, no cross-validation, 70/30 train/test split")
    print("ACCURACY_BY_MAJORITY: ~{}%".format(round(majority_accuracy*100, 2)))
    print("ACCURACY: ~{}%".format(round(accuracy*100, 2)))

def test2(save=False):
    corpus = read_corpus('data/training_data.txt')
    classes = [l['genre'] for l in corpus]
    texts = [l['sentence'] for l in corpus]

    num_folds = 10
    # majority_accuracy = cross_validate_majority(make_mcc, classes, texts)
    avg_accuracy = cross_validate(make_mcc, classes, texts, num_folds)

    if save:
        t = time.strftime("%d-%m-%Y")
        model.save('models/mcc_model_%s.trn' % (t), 'models/mcc_config_%s.json' % (t))

    print("Task 1, neural net, num of cross-validation folds: {}".format(num_folds))
    print("ACCURACY: ~{}%".format(round(avg_accuracy*100, 2)))

def test3():
    corpus = read_corpus('data/training_data.txt')
    classes = [l['polarity'] for l in corpus]
    texts = [l['sentence'] for l in corpus]

    num_folds = 2
    # majority_accuracy = cross_validate_majority(make_mcc, classes, texts)
    avg_accuracy = cross_validate(make_mcc, classes, texts, num_folds)

    print("Task 2, neural net, num of cross-validation folds: {}".format(num_folds))
    print("ACCURACY: ~{}%".format(round(avg_accuracy*100, 2)))

def test4():
    corpus = read_corpus('data/training_data.txt')
    classes = [l['topic'] for l in corpus]
    texts = [l['sentence'] for l in corpus]

    num_folds = 10
    majority_accuracy = cross_validate_majority(make_mcc, classes, texts)
    avg_accuracy = cross_validate(make_mcc, classes, texts, num_folds)

    print("Task 3, neural net, num of cross-validation folds: {}".format(num_folds))
    print("ACCURACY: ~{}%".format(round(avg_accuracy*100, 2)))

def task_one_model():
    corpus = read_corpus('data/training_data.txt')
    classes = [l['genre'] for l in corpus]
    texts = [l['sentence'] for l in corpus]

    num_folds = 10
    majority_accuracy = cross_validate_majority(make_mcc, classes, texts)
    avg_accuracy = cross_validate(make_mcc, classes, texts, num_folds)

    print("Task 3, neural net, num of cross-validation folds: {}".format(num_folds))
    print("ACCURACY: ~{}%".format(round(avg_accuracy*100, 2)))

def make_mcc(classes, texts):
    return MultiClassClassifier(classes, texts)

def make_best_model(model_maker, class_key, num_splits=10):

    corpus = read_corpus('data/training_data.txt')
    classes = [l[class_key] for l in corpus if l['topic'] != "NONE"]
    texts = [l['sentence'] for l in corpus if l['topic'] != "NONE"]

    kfold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=42)
    accuracy_scores = 0

    classes = np.asarray(classes)
    texts = np.asarray(texts)

    max_accuracy_score = 0
    max_model = ''
    count = 0

    for train, test in kfold.split(texts, classes):
        model = model_maker(classes[train], texts[train])
        accuracy = model.evaluate(classes[test], texts[test])
        if accuracy > max_accuracy_score:
            max_accuracy_score = accuracy
            max_model = model
        t = time.strftime("%d-%m-%Y")
        model.save('models/mcc_model_%i_%s.h5' % (count, t), 'models/mcc_config_%i_%s.json' % (count, t))
        count += 1

    max_model.save('models/mcc_model_best_%s.h5' % (t), 'models/mcc_config_best_%s.json' % (t))


def model_from_file(config_path, weights_path):

    model = MultiClassClassifier()
    model.load_from_file(config_path, weights_path)

    # sanity check
    # test = ["It was also the right balance of war and love."]
    # genre = "GENRE_B" #genre

    test = ["I believe so because I was shot from less than 20 metres."]
    genre = "(FEAR_OF)_PHYSICAL_PAIN" #topic
    print(model.predict_sentence(test)==genre)

def make_model_for_class(class_key, save_path=''):
    """

    :param class_key: 'genre', 'topic', 'polarity'
    """
    corpus = read_corpus('data/training_data.txt')
    classes = [l[class_key] for l in corpus]
    texts = [l['sentence'] for l in corpus]

    model = MultiClassClassifier(classes, texts)

    if save_path == '':
        t = time.strftime("%d-%m-%Y")
        model.save('models/mcc_model_%s.h5' % (t), 'models/mcc_config_%s.json' % (t))

    return model

if __name__ == "__main__":

    make_best_model(make_mcc, 'topic', 3)
    # make_model_for_class('polarity')

    # model_from_file('models/mcc_config_04-11-2017.json', 'models/mcc_model_04-11-2017.h5')
    # run pretrained demo
    # model = model_from_json('models/mcc_config_###.json')
    # model.load_weights('models/mcc_model_###.json')
    # demo(model)

    # task 1, LogisticRegressionBinaryClassifier, 10 fold split
    # test0()

    # task 1, MultiClassClassifier, 60/40 split
    # test1(save=True)

    # task 1, MultiClassClassifier, 10 fold split
    # test2()

    # task 2, MultiClassClassifier, 2 fold split
    # test3()

    # task 3, MultiClassClassifier, 10 fold split
    # test4()
