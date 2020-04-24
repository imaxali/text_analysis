import os
import sys
import nltk
import pymorphy2
import csv
from functools import reduce
import math

sys.path.append('lab1')
sys.path.append('lab2')
from re_tokenize import Tokenization
from converting_to_xml import XMLConverter as converter


def create_dict():
    print('Type in sample thematics:')
    sample_name = input().lower()

    global tokens, stems, frq
    frq = 1
    files = os.listdir('lab3/samples/%s' % sample_name)
    common_voc = dict()

    lemmatizer = pymorphy2.MorphAnalyzer()
    stemmer = nltk.stem.SnowballStemmer('russian')

    csv_main = csv.writer(open('lab3/dicts/%s/words_weights.csv' % sample_name, 'w+', encoding='utf8', newline=''))
    csv_main.writerow(['word', 'tf/idf'])

    csv_stops = csv.writer(open('lab3/dicts/%s/stop_words.csv' % sample_name, 'w+', encoding='utf8', newline=''))
    csv_stops.writerow(['stop-word'])

    doc_num_have_word = dict()

    for fl in files:
        markup = converter().convert('lab3/samples/%s/' % sample_name, fl)

        tokens = Tokenization().tokenize(markup)
        lemmas = [lemmatizer.parse(tk)[0].normal_form for tk in tokens]
        stems = [stemmer.stem(l) for l in lemmas]

        voc = dict()

        for idx, stem in enumerate(stems):
            if stem not in voc:
                frq = stems.count(stem)
                voc[stem] = frq

        if not bool(common_voc):
            common_voc = {**voc}
        else:
            for k1 in voc:
                for i, k2 in enumerate(list(common_voc)):
                    if k1 == k2:
                        common_voc[k2] += voc[k2]
                        if doc_num_have_word.get(k2):
                            doc_num_have_word[k2] += 1
                        else:
                            doc_num_have_word[k2] = 2
                        break
                    elif i == len(common_voc) - 1:
                        common_voc[k1] = voc[k1]

    for k in list(common_voc):
        max_frq = reduce(lambda p, c: max(p, c), [v for k, v in common_voc.items()])
        tf = 0.5 + 0.5 * common_voc[k] / max_frq
        common_voc[k] = tf

        idf = math.log(len(files) / (doc_num_have_word.get(k) if doc_num_have_word.get(k) else 1))
        common_voc[k] = float('%.3f' % (common_voc[k] * idf))
        if common_voc[k] < 0.25:
            csv_stops.writerow([k])
            common_voc.pop(k)

    common_voc = {k: v for k, v in sorted(common_voc.items(), key=lambda el: el[1], reverse=True)}

    for k in common_voc:
        csv_main.writerow([k, common_voc[k]])


print(create_dict())
