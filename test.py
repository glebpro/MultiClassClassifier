

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


def test0():

    corpus = read_corpus('data/training_data.txt')
    classes = [l['topic'] for l in corpus if l['topic'] != "NONE"]
    texts = [l['sentence'] for l in corpus if l['topic'] != "NONE"]

    # 80/20 train/test split
    split = int(len(classes) * .8)

    training_classes = classes[:split]
    training_texts = texts[:split]
    test_classes = classes[split:]
    test_texts = classes[split:]

    epochs = 1

    mcc = MultiClassClassifier('data/glove.6b.100d.txt', training_classes, training_texts, epochs)
    accuracy = mcc.evaluate(test_classes, test_texts)

    print(accuracy)

test0()
