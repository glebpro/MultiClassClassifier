
#
#   Tests for MultiClassClassifier and task 3
#
#   Test with 80/20 split
#
#   Cross validate 5 splits
#

import time
from sklearn.model_selection import StratifiedKFold
import numpy as np

from classifiers.MultiClassClassifier import MultiClassClassifier

def read_corpus(fname):
    result = []
    with open(fname) as f:
        for line in f:
            line = line.strip().split("\t")
            result.append({
                "id": line[0].replace(' ', ''),
                "sentence": line[1].replace(' ', ''),
                "genre": line[4].replace(' ', ''),
                "polarity": line[2].replace(' ', ''),
                "topic": line[3].replace(' ', '')
            })
    return result


def test1():

    corpus = read_corpus('data/training_data.txt')
    classes = [l['topic'] for l in corpus if l['topic'] != "NONE"]
    texts = [l['sentence'] for l in corpus if l['topic'] != "NONE"]

    # 80/20 test split
    split = int(.2 * len(texts))
    train_classes = classes[split:]
    train_texts = texts[split:]
    test_classes = classes[:split]
    test_texts = texts[:split]

    epochs = 10
    glove_vectors_path = 'data/glove.6B.100d.txt'

    mlc = MultiClassClassifier(glove_vectors_path, train_classes, train_texts, epochs)
    acc = mlc.evaluate(test_classes, test_texts)
    print(acc)
    # t = time.strftime("%d-%m-%Y")
    # mlc.save('models/1/%s/mcc_unigrams_model_%i_%s.h5' % (i[1], i[0], t),
    #            'models/1/%s/mcc_unigrams_config_%i_%s.json' % (i[1], i[0], t))
    # with open('models/1/%s/config.txt' % i[1], 'w+') as config:
    #     config.write('test/train split: 80/20')
    #     config.write('\nunigrams count')
    #     config.write('\nepochs: ' + str(i[0]))
    #     config.write('\naccuracy: ' + str(acc))

test1()

def test2():
    kfold = StratifiedKFold(n_splits=2, shuffle=False, random_state=42)
    accuracy_scores = 0

    corpus = read_corpus('../../data/training_data.txt')
    classes = [l['topic'] for l in corpus if l['topic'] != "NONE"]
    texts = [l['sentence'] for l in corpus if l['topic'] != "NONE"]

    classes = np.asarray(classes)
    texts = np.asarray(texts)

    for i in [(1, 'a'), (2, 'b'), (3, 'c'), (5, 'd')]:

        max_accuracy_score = 0
        max_model = ''
        split_count = 0

        for train, test in kfold.split(texts, classes):
            mcc = MultiClassClassifier(classes[train], texts[train], i[0])
            accuracy = mcc.evaluate(classes[test], texts[test])
            print(accuracy)
            if accuracy > max_accuracy_score:
                max_accuracy_score = accuracy
                max_model = mcc
            t = time.strftime("%d-%m-%Y")
            mcc.save('models/2/%s/mcc_unigrams_split%s_model_%i_%s.h5' % (i[1], split_count, i[0], t),
                       'models/2/%s/mcc_unigrams_split%s_config_%i_%s.json' % (i[1], split_count, i[0], t))

            with open('models/2/%s/config%s.txt' % (i[1], split_count), 'w+') as config:
                config.write('cross-splits: 2')
                config.write('\nunigrams count')
                config.write('\nepochs: ' + str(i[0]))
                config.write('\naccuracy: ' + str(accuracy))

            split_count += 1


# test2()

#   3) 4 splits
#       a. 1 epoch
#       b. 5 epochs
#       c. 10 epochs
#       d. 25 epochs
