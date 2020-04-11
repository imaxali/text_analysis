import numpy as np
import abc
import math
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import gensim

import re
from re_tokenize import single_words


str1 = 'Lorem ipsum dolor cit amet.'
str2 = 'Loram ipsUm dolor cit aet'


class IStrComparison(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subcls):
        return (
            hasattr(subcls, 'perform_comparison') and
            callable(subcls.perform_comparison)
        )


class LevenshteinDistanceRatio:
    def perform_comparison(self, a: str, b: str) -> float:
        rows = len(a) + 1
        cols = len(b) + 1
        distance = np.zeros((rows, cols), dtype=int)

        for i in range(1, rows):
            for k in range(1, cols):
                distance[i][0] = i
                distance[0][k] = k
        for col in range(1, cols):
            for row in range(1, rows):
                if a[row - 1] == b[col - 1]:
                    cost = 0
                else:
                    cost = 2
                distance[row][col] = min(distance[row - 1][col] + 1,
                                         distance[row][col - 1] + 1,
                                         distance[row - 1][col - 1] + cost)

        ratio = (len(str1) + len(str2) - distance[row][col]) / (len(str1)+len(str2))
        return float('%.3f' % ratio)


class JaroWinklerSimilarity:
    def perform_comparison(self, a: str, b: str, p: float = 0.1, max_l=None) -> float:
            jaro_sim = self.jaro_similarity(a, b)
            max_l = max_l if max_l else len(a)

            prefix_ln = 0
            for i in range(len(a)):
                if a[i] == b[i]:
                    prefix_ln += 1
                else:
                    break
                if prefix_ln == max_l:
                    break

            similarity = jaro_sim + (prefix_ln * p * (1 - jaro_sim))
            return float('%.3f' % similarity)

    @staticmethod
    def jaro_similarity(a: str, b: str) -> float:
            a_ln = len(a)
            b_ln = len(b)
            match_bound = math.floor(max(a_ln, b_ln) / 2) - 1

            matches = 0
            transpositions = 0
            for ch1 in a:
                if ch1 in b:
                    pos1 = a.index(ch1)
                    pos2 = b.index(ch1)
                    if abs(pos1 - pos2) <= match_bound:
                        matches += 1
                        if pos1 != pos2:
                            transpositions += 1

            if matches == 0:
                return 0
            else:
                return 1 / 3 * (
                        matches / a_ln +
                        matches / b_ln +
                        (matches - transpositions // 2) / matches
                )


class HammingDistanceCoef:
    def perform_comparison(self, a: str, b: str) -> float:
        max_ln = max(len(a), len(b))
        return float(
            '%.3f' % (
                1 - (
                    sum(s1 != s2 for s1, s2 in zip(a, b)) +
                    max_ln - min(len(a), len(b))
                )
                / max_ln
            )
        )


def build_hierarchy():
    model = gensim.models.Word2Vec([single_words], min_count=1)

    link_model = linkage(model.wv.syn0, method='complete')

    plt.figure(figsize=(12, 12))
    plt.title('Дендрограмма иерархической кластеризации')
    plt.ylabel('Расстояние')
    plt.xlabel('Слова')

    dendrogram(
        link_model,
        leaf_font_size=6,
        orientation='top',
        leaf_label_func=lambda v: str(model.wv.index2word[v])
    )

    plt.show()


print(LevenshteinDistanceRatio().perform_comparison(str1, str2))
print(JaroWinklerSimilarity().perform_comparison(str1, str2))
print(HammingDistanceCoef().perform_comparison(str1, str2))
print(build_hierarchy())
