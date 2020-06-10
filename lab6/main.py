import sys
from scipy.sparse.linalg import svds
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cosine


sys.path.append('lab4')
from vect_models import Estimation


class PlsaVSWord2vec:
    def __init__(self, k, v_m):
        self.k = k
        self.vec_mods = v_m
        self.m = v_m.doc_term_m
        self.plsa = []
        self.w2v = []

    def normalize_series(self, series):
        return (series - series.min()) / (series.max() - series.min())

    def normalize_df(self, df):
        res = df.copy()
        for d in df.columns:
            max_val = res[d].max()
            min_val = res[d].min()
            res[d] = (df[d] - min_val) / max_val - min_val
        return res

    def compute_plsa_m(self):
        doc_count = len(self.m.toarray()[0])
        word_count = len(self.m.toarray())
        doc_topic_prob = self.normalize_df(pd.DataFrame(np.random.random(
            size=(doc_count, self.k)),
            index=[i for i in range(doc_count)]))
        topic_word_prob = self.normalize_df(pd.DataFrame(np.random.random(
            size=(self.k, word_count)),
            columns=[i for i in range(word_count)]))
        topic_prob = np.zeros([doc_count, word_count, self.k])

        rows, cols = self.m.nonzero()

        for w_i in rows:
            for d_i in cols:
                prob = doc_topic_prob.loc[d_i, :] * topic_word_prob.loc[:, w_i]
                topic_prob[d_i][w_i] = self.normalize_series(prob)
                print(topic_prob[d_i][w_i])

        for topic in range(self.k):
            for _w_i in rows:
                res = 0
                for _d_i in cols:
                    res += self.m.toarray()[_w_i][_d_i] * topic_prob[_d_i, _w_i, topic]
                topic_word_prob.loc[topic][_w_i] = res
            topic_word_prob.loc[topic] = self.normalize_series(topic_word_prob.loc[topic])
        for _d_i in range(doc_count):
            for topic in range(self.k):
                res = 0
                for _w_i in range(word_count):
                    res += self.m.toarray()[_w_i][_d_i] * topic_prob[_d_i, _w_i, topic]
                topic_word_prob.loc[_d_i][topic] = res
            topic_word_prob.loc[_d_i] = self.normalize_series(topic_word_prob.loc[_d_i])
        return topic_word_prob

    def compute_word2vec_m(self, d_t):
        df = [[j for i in [k for k in v] for j in i] for v in d_t]
        # df = itertools.chain.from_iterable(d_t)
        model = Word2Vec(df, min_count=1, window=20, sg=1)
        self.w2v = self.normalize_df(pd.DataFrame(model.wv[model.wv.vocab.keys()]).T)

    def compute_sim(self, m):
        print('Cosine sim: {}'.format(self.vec_mods.cosine_sim(m, 'drink', 'car')))
        print('Jaccard sim: {}'.format(self.vec_mods.jaccard_sim(m, 'drink', 'car')))


if __name__ == '__main__':
    vec_mods = Estimation()
    vec_mods.get_matrices()

    vec_mods2 = PlsaVSWord2vec(5, vec_mods)
    print(vec_mods2.compute_plsa_m())
    print(vec_mods2.compute_word2vec_m(vec_mods.sentences))
    # print(vec_mods.pearson_corrcoef(vec_mods2.w2v, vec_mods2.plsa))
    print(vec_mods2.compute_sim(vec_mods2.w2v))
    print(vec_mods2.compute_sim(vec_mods2.plsa))

    #                       Wordsim353  Cosine sim  Jaccard sim
    #   tiger,tiger         10          10          10
    #   tiger,cat           7.35        3.17        1.2
    #   plane,car           5.77        1.91        1.2
    #   car,jaguar          7.27        1.91        1.2
    #   tiger,jaguar        8           8.43        4.57
    #   cat,jaguar          7.42        8.43        4.57
    #   drink,car           3.04        1.91        1.2
    #   mother,drink        2.65        0.98        0.65
    #   doctor,professor    6.62        5.94        2.07
    #   doctor,nurse        7           5.94        2.07
