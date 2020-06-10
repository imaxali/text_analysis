import os
import sys
import nltk
import pymorphy2
import csv
from functools import reduce
import math
from scipy import sparse

sys.path.append('lab1')
sys.path.append('lab2')
from re_tokenize import Tokenization
from converting_to_xml import XMLConverter as converter


class Vectorisation:
    def __init__(self):
        self.sample_name = None
        self.common_voc, self.df = {}, {}
        self.files, self.doc_term_matrix, self.tf_idf_matrix, self.doc_term, self.sentences = \
            [], [], [], [], []

    def filter_stops(self):
        csv_main = csv.writer(
            open(
                'lab3/dicts/%s/words_weights.csv' % self.sample_name,
                'w+',
                encoding='utf8',
                newline=''
            )
        )
        csv_main.writerow(['word', 'tf/idf'])

        csv_stops = csv.writer(
            open(
                'lab3/dicts/%s/stop_words.csv' % self.sample_name,
                'w+',
                encoding='utf8',
                newline=''
            )
        )
        csv_stops.writerow(['stop-word'])

        for k in list(self.common_voc):
            if self.common_voc[k] < 0.25:
                csv_stops.writerow([k])
                self.common_voc.pop(k)

        sorted_voc = {k: v for k, v in sorted(self.common_voc.items(), key=lambda el: el[1], reverse=True)}

        for k in sorted_voc:
            csv_main.writerow([k, sorted_voc[k]])

    def tf_idf_counter(self):
        for w in self.doc_term_matrix.toarray():
            self.tf_idf_matrix.append([0]*(len(w)))
        for idx in range(len(self.files)):
            max_frq = reduce(lambda p, c: max(p, c), [w[idx] for w in self.doc_term_matrix.toarray()])
            for i, w, k in zip(range(len(self.doc_term_matrix.data)), self.doc_term_matrix.toarray(), self.common_voc):
                if w[idx] != 0:
                    tf = 0.5 + 0.5 * w[idx] / max_frq
                    idf = math.log(len(self.files) / (self.df.get(k) if self.df.get(k) else 1))
                    self.tf_idf_matrix[i][idx] = float('%.3f' % (tf * idf))
        self.tf_idf_matrix = sparse.lil_matrix(self.tf_idf_matrix)

    def df_counter(self, sample=None):
        if sample is None:
            print('Type in sample thematics:')
            self.sample_name = 'lab3/samples/' + input().lower()
        else:
            self.sample_name = sample

        self.files = os.listdir(self.sample_name)

        lemmatizer = pymorphy2.MorphAnalyzer()
        stemmer = nltk.stem.SnowballStemmer('russian') if __name__ == 'main' else nltk.stem.porter.PorterStemmer()

        for fl in self.files:
            markup = converter().convert(fl, self.sample_name)

            tokens, sentences = Tokenization().tokenize(markup)
            lemmas = [lemmatizer.parse(tk)[0].normal_form for tk in tokens]
            stems = [stemmer.stem(l) for l in lemmas]

            sent_lemmas = [[lemmatizer.parse(tk)[0].normal_form for tk in sentence] for sentence in sentences]
            sent_stems = [[stemmer.stem(l) for l in sentence] for sentence in sent_lemmas]

            self.sentences.append(sent_stems)
            voc = dict()

            for idx, stem in enumerate(stems):
                if stem not in voc:
                    frq = stems.count(stem)
                    voc[stem] = frq

            self.doc_term.append(voc)

            if not bool(self.common_voc):
                self.common_voc = {**voc}
            else:
                for k1 in voc:
                    for i, k2 in enumerate(list(self.common_voc)):
                        if k1 == k2:
                            self.common_voc[k2] += voc[k2]
                            if self.df.get(k2):
                                self.df[k2] += 1
                            else:
                                self.df[k2] = 2
                            break
                        elif i == len(self.common_voc) - 1:
                            self.common_voc[k1] = voc[k1]

        self.doc_term_matrix = []
        for k in self.common_voc:
            self.doc_term_matrix.append([0] * len(self.files))
            for i, v in zip(range(len(self.doc_term)), self.doc_term):
                if v.get(k):
                    self.doc_term_matrix[len(self.doc_term_matrix) - 1][i] = v[k]
        self.doc_term_matrix = sparse.lil_matrix(self.doc_term_matrix, dtype=float)

    def contrast_selection(self, samples):
        dicts = []
        for sample in samples:
            with open('lab3/dicts/%s/words_weights.csv' % sample, encoding='utf8') as csvf:
                csvr = csv.DictReader(csvf)
                dicts.append({})
                for row in csvr:
                    if float(row['tf/idf']) >= 1.382:
                        dicts[len(dicts) - 1][row['word']] = row['tf/idf']
                    else:
                        break
        for i in range(len(dicts) - 1):
            for k in list(dicts[i]):
                for j in range(i + 1, len(dicts)):
                    if k in dicts[j]:
                        dicts[i].pop(k)
                        dicts[j].pop(k)
                        print(k)
        for i in range(len(samples)):
            with open('lab3/dicts/%s/key_words.csv' % samples[i], 'w+', encoding='utf8', newline='') as f:
                csvw = csv.writer(f)
                csvw.writerow(['keyword'])
                for k in dicts[i]:
                    csvw.writerow([k])


if __name__ == '__main__':
    dicts = Vectorisation()
    print(dicts.df_counter())
    # print(dicts.tf_idf_counter())
    # print(dicts.tf_idf_matrix)
