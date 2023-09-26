"""
Label encoding transforms
"""

import torch
import pickle
from hierarchy.tree import DistanceDict


class SoftLabels:
    def __init__(self, cfg):
        with open(cfg.dist_path, "rb") as f:
            self.distances = DistanceDict(pickle.load(f))
        self.classes = cfg.DATASET['CLASS_ABBR']
        self.all_labels = self.encode_all_labels(cfg)

    def encode_all_labels(self, cfg):
        hardness = cfg.soft_beta
        distance_matrix = torch.Tensor([[self.distances[c1, c2] for c1 in self.classes] for c2 in self.classes])
        max_distance = torch.max(distance_matrix)
        distance_matrix /= max_distance
        labels = torch.exp(-hardness * distance_matrix) / torch.sum(torch.exp(-hardness * distance_matrix), dim=0)
        return labels

    def encode_batch_labels(self, y: torch.Tensor):
        batch_size = y.shape[0]
        labels = torch.zeros((batch_size, len(self.classes)), device=y.device, dtype=torch.float)
        for i in range(batch_size):
            this_label = self.all_labels[:, y[i]]
            labels[i, :] = this_label
        return labels


class MLB:
    def __init__(self, cfg):
        from sklearn.preprocessing import MultiLabelBinarizer
        self.encoder = MultiLabelBinarizer(classes=cfg.DATASET['CLASS_SET'])
        self.all_labels = self.encode_all_labels(cfg)

    def encode_all_labels(self, cfg): # fit
        y = [set(name.split('/')) for name in cfg.DATASET['CLASS_LONG']]
        labels = self.encoder.fit_transform(y)
        return torch.from_numpy(labels)

    def encode_batch_labels(self, y: torch.Tensor): # transform
        return self.all_labels[y].cuda()

    # def revert(self, y: torch.Tensor):
    #     return y.argmax(-1)


class OneHot:
    def __init__(self, cfg):
        self.nclasses = cfg.nclasses['all']

    def encode_batch_labels(self, y: torch.Tensor):
        """ 1-hot encodes a tensor to another similarly stored tensor"""
        if len(y.shape) > 0 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        out = torch.zeros(y.size()+torch.Size([self.nclasses]), device=y.device, dtype=torch.float)
        return out.scatter_(-1, y.view((*y.size(), 1)), 1)

    def revert(self, y: torch.Tensor):
        return y.argmax(-1)

