from collections import defaultdict
import math
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

    def classify(self, classifier, test_sample):
        classes, prob = classifier
        predicts = []
        for feats in test_sample:
            predict = min(classes.keys(),
                          key=lambda cl: -math.log(classes[cl]) +
                          sum(-math.log(prob.get((cl, feat), 10**(-7))) for feat in feats[1]))
            predicts.append(predict)
        return predicts


if __name__ == '__main__':
    dataset = list(csv.reader(open('lab5/spam.csv', 'r', encoding='ISO-8859-1')))
    dataset = [(line[0], ''.join(line[1:])) for line in dataset]
    dataset = [(line[0], [w.lower() for w in Tokenization().tokenize(line[1], False)]) for line in dataset]

    stemmer = nltk.stem.porter.PorterStemmer()

    dataset = [(line[0], [stemmer.stem(w) for w in line[1]]) for line in dataset]

    threshold = 0.5
    prev_recall = 0
    avg_precision = 0
    while threshold > 0:
        train_set, test_set = train_test_split(dataset, test_size=threshold, random_state=1)

        classifier = BayesClassifier().train(train_set)

        get_predicts = BayesClassifier().classify(classifier, test_set)
        estimation = confusion_matrix([sms[0] for sms in test_set], get_predicts)

        precision = estimation[1][1] / (estimation[1][1] + estimation[1][0])
        recall = estimation[1][1] / (estimation[1][1] + estimation[0][1])
        avg_precision += (recall - prev_recall) * precision
        prev_recall = recall
        threshold = float('%.2f' % threshold) - 0.1

    avg_precision = float('%.2f' % avg_precision)
    print(avg_precision)
    # 0.8 - spam as positive & 0.97 - ham as positive

    lab3 = Vectorisation()
    lab3.df_counter()
    dicts = lab3.dicts_terms
    dicts = [[k for k in dic] for dic in dicts]
    dicts_predicts = BayesClassifier().classify(classifier, dicts)
    print(dicts_predicts)
