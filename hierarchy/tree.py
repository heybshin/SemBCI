import pickle
import numpy as np
from math import exp, fsum
from nltk.tree import Tree
from copy import deepcopy


class DistanceDict(dict):
    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}
        self.max_dist = max(distances.values())

    def __getitem__(self, i):
        if i[0] == i[1]:
            return 0
        else:
            return self.distances[(i[0], i[1]) if i[0] < i[1] else (i[1], i[0])]

    def __setitem__(self, i):
        raise NotImplementedError()


def get_distances_similarities(cfg):
    with open(cfg.DATA.dist_path, "rb") as f:
        distances = DistanceDict(pickle.load(f))

    classes = cfg.DATA.class_abbr
    distance_matrix = np.zeros([len(classes), len(classes)])
    best_hiersim = np.zeros([len(classes), len(classes)])

    for i in range(len(classes)):
        for j in range(len(classes)):
            distance_matrix[i, j] = distances[(classes[i], classes[j])]

    for i in range(len(classes)):
        best_hiersim[i, :] = 1 - np.sort(distance_matrix[i, :]) / distances.max_dist

    return distances, best_hiersim


def get_weighting(hierarchy: Tree, alpha, normalize=True):
    """
    Construct exponentially decreasing weighting, where each edge is weighted
    according to its distance from the root as exp(-alpha*dist).

    Args:
        hierarchy: The tree to generate the weighting for.
        alpha: The decay alpha.
        normalize: If True ensures that the sum of all weights sums
            to one.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    all_weights = []
    for p in weights.treepositions():
        node = weights[p]
        weight = exp(-alpha * len(p))
        all_weights.append(weight)
        if isinstance(node, Tree):
            node.set_label(weight)
        else:
            weights[p] = weight
    total = fsum(all_weights)  # stable sum
    if normalize:
        for p in weights.treepositions():
            node = weights[p]
            if isinstance(node, Tree):
                node.set_label(node.label() / total)
            else:
                weights[p] /= total

    return weights