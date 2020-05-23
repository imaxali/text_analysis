from collections import defaultdict
import math
import pandas as pd
import csv
import sys
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.append('lab2')
sys.path.append('lab3')

from re_tokenize import Tokenization
from voc_compose import Vectorisation


class BayesClassifier:
    def train(self, samples):
        classes,frq = defaultdict(lambda: 0), defaultdict(lambda: 0)
        for lbl, feats in samples:
            classes[lbl] += 1
            for feat in feats:
                frq[lbl, feat] += 1
        for lbl, feat in frq:
            frq[lbl, feat] /= classes[lbl]
        for c in classes:
            classes[c] /= len(samples)
        return classes, frq

    def classify(self, classifier, test_sample, th):
        classes, prob = classifier
        predicts = []
        prob_classes_sum = {}
        for feats in test_sample:
            for cl in classes:
                prob_sum = -math.log(classes[cl]) +\
                    sum(-math.log(prob.get((cl, feat), 10**(-7))) for feat in feats[1])
                prob_classes_sum[cl] = prob_sum
            if prob_classes_sum['ham'] > prob_classes_sum['spam'] >= th:
                predicts.append('spam')
            else:
                predicts.append('ham')
        return predicts


if __name__ == '__main__':
    dataset = list(csv.reader(open('lab5/spam.csv', 'r', encoding='ISO-8859-1')))
    dataset.pop(0)
    dataset = [(line[0], ''.join(line[1:])) for line in dataset]
    dataset = [(line[0], [w.lower() for w in Tokenization().tokenize(line[1], False)]) for line in dataset]

    stemmer = nltk.stem.porter.PorterStemmer()

    dataset = [[line[0], [stemmer.stem(w) for w in line[1]]] for line in dataset]

    threshold = 0.1
    prev_recall = 0
    avg_precision = 0
    while threshold > 0:
        train_set, test_set = train_test_split(dataset, test_size=threshold)
        classifier = BayesClassifier().train(train_set)

        get_predicts = BayesClassifier().classify(classifier, test_set, 40)
        estimation = confusion_matrix([sms[0] for sms in test_set], get_predicts)

        precision = estimation[1][1] / (estimation[1][1] + estimation[0][1]) if \
            (estimation[1][1] + estimation[0][1]) \
            else 0
        recall = estimation[1][1] / (estimation[1][1] + estimation[1][0]) if \
            (estimation[1][1] + estimation[1][0]) \
            else 0
        print(estimation)
        print(precision, recall)
        avg_precision += (recall - prev_recall) * precision
        prev_recall = recall
        threshold = float('%.3f' % threshold) - 0.1

    avg_precision = float('%.2f' % avg_precision)
    print(avg_precision)

    # [[473   1]
    #  [4  80]]
    # 0.9876543209876543
    # 0.9523809523809523
