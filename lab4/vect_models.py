import sys
from scipy import sparse
import numpy as np
import math

sys.path.append('lab3')
from voc_compose import Vectorisation


class Estimation:
    tf_idf = []
    doc_term_m = []
    words = {}
    word1_i, word2_i = (0,)*2
    doc_term = []

    def find_word_idx(self, w1, w2):
        for i, k in zip(range(len(self.words)), self.words):
            if k == w1 and w1 == w2:
                self.word1_i = i
                self.word2_i = i
                break
            elif k == w1:
                self.word1_i = i
                break
            elif k == w2:
                self.word2_i = i
                break
            elif i == len(self.words) - 1:
                raise ValueError('No such word in sample dict')

    def jaccard_sim(self, md, w1, w2):
        self.find_word_idx(w1, w2)
        global max_frq, min_frq
        max_frq, min_frq = 0, 0

        for w1, w2 in zip(md[self.word1_i], md[self.word2_i]):
            max_frq += max(w1, w2)
            min_frq += min(w1, w2)

        # sim = abs(min_frq) / (max_frq - min_frq) if (max_frq - min_frq) != 0 else 1
        sim = min_frq / max_frq
        # intersection = len(set(md.toarray()[self.word1_i]).intersection(set(md.toarray()[self.word2_i])))
        # union = len(set(md.toarray()[self.word1_i]).union(set(md.toarray()[self.word2_i])))
        #
        # return intersection / float(union)
        return float('%.2f' % (sim * 10))

    def cosine_sim(self, md, w1, w2):
        self.find_word_idx(w1, w2)

        dot_product, v1_magnitude, v2_magnitude = (0,) * 3
        for v1, v2 in zip(md[self.word1_i], md[self.word2_i]):
            dot_product += v1 * v2
            v1_magnitude += v1 * v1
            v2_magnitude += v2 * v2

        v1_magnitude = math.sqrt(v1_magnitude)
        v2_magnitude = math.sqrt(v2_magnitude)
        magnitude = v1_magnitude * v2_magnitude

        cos_sim = dot_product / magnitude
        return float('%.2f' % (cos_sim * 10))

    def pearson_corrcoef(self, mod):
        # m = sparse.vstack((mod1, mod2), format='csr', dtype=float)
        # n = m.shape[1]
        #
        # rowsum = m.sum(1)
        # centering = rowsum.dot(rowsum.T.conjugate()) / n
        # cov = (m.dot(m.T.conjugate()) - centering) / (n - 1)
        #
        # diag = np.diag(cov)
        # coeffs = cov / np.sqrt(np.outer(diag, diag))

        coeffs = np.corrcoef(mod)
        return coeffs
        # [[1.         0.68854723 0.74003158 ... 0.09284145 0.14167412 0.12035408]
        # [0.68854723 1.         0.58815689 ... 0.02992828 0.15870603 0.06807386]
        # [0.74003158 0.58815689 1.         ... 0.07068274 0.11477989 0.09273851]
        # ...
        # [0.09284145 0.02992828 0.07068274 ... 1.         0.1184184  0.08872198]
        # [0.14167412 0.15870603 0.11477989 ... 0.1184184  1.         0.09043597]
        # [0.12035408 0.06807386 0.09273851 ... 0.08872198 0.09043597 1.        ]]

        # [[1.          0.08214242  0.38715307    ...     -0.09278419    -0.09278419   -0.09278419]
        #  [0.08214242    1.          0.2386587   ...  0.00856524  0.00856524 0.00856524]
        #  [0.38715307 0.2386587    1.  ...    -0.03448276 -0.03448276 -0.03448276]
        #  ...
        #  [-0.09278419    0.00856524  -0.03448276 ... 1.  1.  1.]
        #  [-0.09278419    0.00856524  -0.03448276 ... 1.  1.  1.]
        #  [-0.09278419    0.00856524  -0.03448276 ... 1.  1.  1.]]

        # [[1.          0.08214242  0.38715307  ... -0.09278419  -0.09278419 -0.09278419]
        #  [0.08214242  1.          0.2386587   ...  0.00856524  0.00856524 0.00856524]
        #  [0.38715307   0.2386587   1.     ...    -0.03448276 -0.03448276 -0.03448276]
        #  ...
        #  [-0.09278419  0.00856524  -0.03448276 ...  1.     1.  1.]
        #  [-0.09278419  0.00856524 - 0.03448276 ...  1.     1.  1.]
        #  [-0.09278419  0.00856524 - 0.03448276 ...  1.     1.  1.]]

    def get_matrices(self):
        dicts = Vectorisation()
        dicts.df_counter('lab4/wordsim_sample/')
        dicts.tf_idf_counter()
        self.doc_term_m = dicts.doc_term_matrix
        self.tf_idf = dicts.tf_idf_matrix
        self.words = dicts.common_voc
        self.doc_term = dicts.doc_term
        self.sentences = dicts.sentences


if __name__ == '__main__':
    vect_model = Estimation()
    print(vect_model.get_matrices())
    print(vect_model.pearson_corrcoef(vect_model.tf_idf))
    print(vect_model.cosine_sim(vect_model.tf_idf, 'doctor', 'nurse'))
    print(vect_model.jaccard_sim(vect_model.tf_idf, 'doctor', 'nurse'))

#   Comparing coeffs similarity metrics of doc-term models with wordsim353
#                       Wordsim353  Cosine Sim  Jaccard Sim
#   tiger,tiger         10          10          10
#   tiger,cat           7.35        5.56        0.7
#   plane,car           5.77        3.46        0.39
#   car,jaguar          7.27        3.46        0.39
#   tiger,jaguar        8           4.12        0.39
#   cat,jaguar          7.42        4.12        0.39
#   drink,car           3.04        3.46        0.39
#   mother,drink        2.65        2.48        0.27
#   doctor,professor    6.62        4.48        0.56
#   doctor,nurse        7           4.48        0.56
