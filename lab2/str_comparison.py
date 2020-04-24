import numpy as np
import abc
import math
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

from lab2.re_tokenize import single_words


class IStrComparison(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subcls):
        return (
            hasattr(subcls, 'comparing') and
            callable(subcls.comparing)
        )


class LevenshteinDistanceRatio:
    def comparing(self, a, b=None) -> float:
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

        ratio = (len(a) + len(b) - distance[row][col]) / (len(a)+len(b))
        return float('%.3f' % ratio)


class JaroWinklerSimilarity:
    def comparing(self, a: str, b: str, p: float = 0.1, max_l=None) -> float:
            sim = self.jaro_similarity(a, b)
            max_l = max_l if max_l else len(a)

            prefix_ln = 0
            for i in range(len(a)):
                try:
                    if a[i] == b[i]:
                        prefix_ln += 1
                    else:
                        break
                except IndexError:
                    break
                if prefix_ln == max_l:
                    break

            res = sim + (prefix_ln * p * (1 - sim))
            return float('%.3f' % res)

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
    def comparing(self, a: str, b: str) -> float:
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


class HierarchicalClustering:
    def distribution(self, a, b=None, alg=1):
        if b is None:
            p1, p2 = a
        if alg == 1:
            if b is None:
                return LevenshteinDistanceRatio().comparing(single_words[p1], single_words[p2])
            return LevenshteinDistanceRatio().comparing(a, b)
        elif alg == 2:
            if b is None:
                return JaroWinklerSimilarity().comparing(single_words[p1], single_words[p2])
            return JaroWinklerSimilarity().comparing(a, b)
        elif alg == 3:
            if b is None:
                return HammingDistanceCoef().comparing(single_words[p1], single_words[p2])
            return HammingDistanceCoef().comparing(a, b)
        else:
            return 'No algorithm by this num'

    def build(self):
        upper_triangle = np.triu_indices(len(single_words), 1)
        w_dist = np.apply_along_axis(self.distribution, 0, upper_triangle, None, 1)

        link_model = linkage(w_dist, method='average')
        plt.figure(figsize=(12, 12))
        plt.title('Дендрограмма иерархической кластеризации')
        plt.ylabel('Расстояние')
        plt.xlabel('Слова')

        dendrogram(
            link_model,
            leaf_font_size=6,
            orientation='top',
            leaf_label_func=lambda v: single_words[v]
        )

        plt.show()


print(HierarchicalClustering().build())

# For execution without building hierarchical clustering
# s1 = 'CHEESE CHORES GESE GLOVES'
# s2 = 'CHESE CORES GEESE GLOVE'
#
# print('Levenstein:', HierarchicalClustering().distribution(1, s1, s2))
# print('JaroWinkler:', HierarchicalClustering().distribution(2, s1, s2))
# print('HammingDistance:', HierarchicalClustering().distribution(3, s1, s2))
