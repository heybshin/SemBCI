"""
Adapted from S. Zhang, R. Xu, C. Xiong, and C. Ramaiah, "Use all the labels: A hierarchical multi-label contrastive learning framework,"
in Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 16,660–16,669.
"""

import torch
import torch.nn as nn
from .Contrastive import SupConLoss

from typing import List
from nltk.tree import Tree
import math


class HiCon(nn.Module):
    def __init__(self,
                 n_epochs,
                 temperature=0.07,
                 base_temperature=0.07,
                 penalty='-al',
                 alpha=1,
                 contrast=False):
        super(HiCon, self).__init__()
        self.n_epochs = n_epochs
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.penalty = penalty
        self.alpha = alpha
        self.sup_con_loss = SupConLoss(temperature, contrast=contrast)

    def pow_2(self, value):
        return torch.pow(2, value)

    def sigmoidal_decay(self, epoch, alphas, k=10):
        midpoint = self.n_epochs / 2
        alpha = alphas[0] + (alphas[1] - alphas[0]) / (1 + math.exp(-k * (epoch - midpoint)))
        return alpha

    def forward(self, features, labels, epoch):
        mask = torch.ones(labels.shape).cuda()
        cumulative_loss = torch.tensor(0.0).cuda()
        max_loss_lower_layer = torch.tensor(float('-inf'))
        if isinstance(self.alpha, list):
            alpha = self.sigmoidal_decay(epoch, self.alpha, k=1)
        else:
            alpha = self.alpha
        max_height = labels.shape[1]
        for l in range(1, max_height):
            mask[:, max_height - l:] = 0  # mask out the last l layers
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).cuda()  # 1 if all labels are equal (positive pair), 0 otherwise
            layer_loss = self.sup_con_loss(features, mask=mask_labels)

            if self.penalty == '-al':
                penalty = - alpha * l
            elif self.penalty == 'div':
                penalty = alpha / l
            else:
                penalty = alpha / (max_height - l)

            cumulative_loss += torch.exp(torch.tensor(penalty).type(torch.float)) * layer_loss
            unique, inverse = torch.unique(layer_labels, sorted=True, return_inverse=True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            unique_indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        return cumulative_loss / labels.shape[1]


class HiCE(torch.nn.Module):
    """
    Hierarchical Cross-Entropy (HiCE) loss, which combines softmax with Hierarchical Log Likelihood Loss.

    The weights must be implemented as an nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shape of the weight tree must be the same as the associated hierarchy.

    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. We use alphabetical ordering on the leaf nodes
    labels, which corresponds to the 'normal' ImageNet ordering.

    Args:
        hierarchy: The hierarchy used to define the loss.
        classes: A list of classes defining the order of the leaf nodes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, hierarchy: Tree, classes: List[str], weights: Tree):
        super(HiCE, self).__init__()
        self.hierarchy = hierarchy
        assert hierarchy.treepositions() == weights.treepositions()

        # the tree positions of all the leaves
        positions_leaves = {c: p for c, p in
                            [(self.get_label(hierarchy[p]), p) for p in hierarchy.treepositions("leaves")] if
                            c in classes}

        # Number of classes
        num_classes = len(positions_leaves)

        # Tree positions of all edges (excluding the root)
        positions_edges = hierarchy.treepositions()[1:]

        # Map from position tuples to leaf/edge indices
        index_map_leaves = {p: i for i, p in enumerate(positions_leaves.values())}
        index_map_edges = {p: i for i, p in enumerate(positions_edges)}

        # Edge indices corresponding to the path from each index to the root
        edges_from_leaf = [[index_map_edges[position[:i]] for i in range(len(position), 0, -1)] for position in
                           positions_leaves.values()]

        # Max size for the number of edges to the root
        num_edges = max(map(len, edges_from_leaf))

        # Indices of all leaf nodes for each edge index
        leaf_indices = [[index_map_leaves[p + leaf] for leaf in self.get_leaf_positions(p) if
                         p + leaf in index_map_leaves] for p in positions_edges]

        # save all relevant information as pytorch tensors for computing the loss on the gpu
        self.onehot_den = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.onehot_num = torch.nn.Parameter(torch.zeros([num_classes, num_classes, num_edges]), requires_grad=False)
        self.weights = torch.nn.Parameter(torch.zeros([num_classes, num_edges]), requires_grad=False)

        # one hot encoding of the numerators and denominators and store weights
        for i in range(num_classes):
            for j, k in enumerate(edges_from_leaf[i]):
                self.onehot_num[i, leaf_indices[k], j] = 1.0
                self.weights[i, j] = self.get_label(weights[positions_edges[k]])
            for j, k in enumerate(edges_from_leaf[i][1:]):
                self.onehot_den[i, leaf_indices[k], j] = 1.0
            self.onehot_den[i, :, j + 1] = 1.0  # the last denominator is the sum of all leaves

    def get_leaf_positions(self, position):
        node = self.hierarchy[position]
        if isinstance(node, Tree):
            return node.treepositions("leaves")
        else:
            return [()]

    @staticmethod
    def get_label(node):
        if isinstance(node, Tree):
            return node.label()
        else:
            return node

    def forward(self, inputs, target):
        """
        inputs: Class logits ordered as the input hierarchy.
        target: The index of the ground truth class.
        """
        if len(target.shape) > 1:
            target = target.argmax(-1)
        inputs = torch.nn.functional.softmax(inputs, 1)  # Applying softmax
        # add a sweet dimension to inputs
        inputs = torch.unsqueeze(inputs, 1) #[640, 1, 10]
        # sum of probabilities for numerators
        num = torch.squeeze(torch.bmm(inputs, self.onehot_num[target]))
        # sum of probabilities for denominators
        den = torch.squeeze(torch.bmm(inputs, self.onehot_den[target]))
        # compute the neg logs for non zero numerators and store in there
        idx = num != 0
        num[idx] = -torch.log(num[idx] / den[idx])
        # weighted sum of all logs for each path (we flip because it is numerically more stable)
        num = torch.sum(torch.flip(self.weights[target] * num, dims=[1]), dim=1)
        # return sum of losses / batch size
        return torch.mean(num)

