"""
Utils for data loading and preprocessing
"""

import random
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transforms import data_transform

from typing import Optional, OrderedDict


def extract_unknown(cfg, data, labels):
    # Extract integer labels for known and unknown classes
    unknown_classes = [cfg.class_long.index(c) for c in cfg.unknown_classes]
    known_classes = [i for i in range(10) if i not in unknown_classes]

    # Create masks for known and unknown classes
    unknown_mask = np.isin(labels[:, -1], unknown_classes)
    known_mask = np.logical_not(unknown_mask)

    # Re-label known classes to be continuous indices from 0
    known_data = data[known_mask]
    new_labels = np.arange(len(known_classes))
    known_labels = new_labels[np.searchsorted(known_classes, labels[known_mask])]

    # Update data config
    cfg.n_classes.all = len(known_classes)
    cfg.class_long = [c for c in cfg.class_long if c not in cfg.unknown_classes]
    # cfg.known_classes = known_classes
    cfg.class_abbr = [c for i, c in enumerate(cfg.class_abbr) if i not in unknown_classes]

    return (known_data, known_labels), EEGDataset(cfg, [data[unknown_mask], labels[unknown_mask]])


def balanced_split(data, labels, frac=0.25, shuffle=True):
    # Do train test split considering class imbalance
    all_idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(all_idx)
    counts = np.round(np.bincount(labels) * frac)
    test_idx = list()
    for i in all_idx:
        if counts[labels[i]] > 0:
            counts[labels[i]] -= 1
            test_idx.append(i)
    test_idx = np.array(test_idx)
    train_idx = np.setdiff1d(all_idx, test_idx)

    return train_idx, test_idx


class DataContainer:
    """
    A container class for holding EEG dataset information.

    Attributes:
    - data (np.ndarray): The EEG data samples.
    - labels (np.ndarray): Corresponding labels for the EEG data samples.
    - train_ids (Optional[np.ndarray]): Indices for the training data.
    - val_ids (Optional[np.ndarray]): Indices for the validation data.
    - unknown_set (Optional[torch.utils.data.Dataset]): Data samples that belong to the open set recognition (OSR) task.

    Example:
    >>> dc = DataContainer(data=my_data, labels=my_labels)
    >>> dc.train_ids = np.array([0, 1, 2, 3])
    >>> dc.val_ids = np.array([4, 5, 6])
    """

    def __init__(self, 
                 data: np.ndarray = None, 
                 labels: np.ndarray = None, 
                 train_ids: Optional[np.ndarray] = None, 
                 val_ids: Optional[np.ndarray] = None, 
                 unknown_set: Optional[torch.utils.data.Dataset] = None):
        self.data = data
        self.labels = labels
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.unknown_set = unknown_set
        

class EEGDataModule(LightningDataModule):
    def __init__(self, cfg, dc: DataContainer):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.batch_size
        self.n_workers = cfg.NUM_WORKERS

        self.data = dc.data
        self.labels = dc.labels
        self.train_ids = dc.train_ids
        self.val_ids = dc.val_ids

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = HierEEGDataset(self.cfg.DATA, [self.data[self.train_ids], self.labels[self.train_ids]])
            self.val = HierEEGDataset(self.cfg.DATA, [self.data[self.val_ids], self.labels[self.val_ids]])
            print(f' Train: {len(self.train)}    Val: {len(self.val)}')

    def train_dataloader(self):
        if self.cfg.SAMPLER == 'hierarchical':
            sampler = HierBatchSampler(self.batch_size, self.train)
        else:
            sampler = None
        return DataLoader(
            self.train, batch_size=1, sampler=sampler,
            num_workers=self.n_workers
        )

    def val_dataloader(self):
        sampler = HierBatchSampler(self.batch_size, self.val)
        return DataLoader(
            self.val, batch_size=1, shuffle=False, sampler=sampler,
            num_workers=self.n_workers, pin_memory=True,
        )

    def predict_dataloader(self):
        dataset = EEGDataset(self.cfg.DATA, [self.data, self.labels[:, -1]])
        return DataLoader(dataset,
                          pin_memory=True,
                          batch_size=100,
                          shuffle=False,
                          num_workers=self.n_workers,
                          persistent_workers=True)


class OSRDataModule(EEGDataModule):
    def __init__(self, cfg, dc: DataContainer):
        super(OSRDataModule, self).__init__(cfg, dc)
        self.unknown = dc.unknown_set

    def val_dataloader(self):
        sampler = HierBatchSampler(self.batch_size, self.val)
        val_loader = DataLoader(
            self.val, batch_size=1, shuffle=False, sampler=sampler,
            num_workers=self.n_workers, pin_memory=True,
        )
        unk_loader = DataLoader(
            self.unknown, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, pin_memory=True,
        )
        return [val_loader, unk_loader]


class EEGDataset(Dataset):
    """
    Dataset of single subject (epoched data)

    Args:
        data: numpy array of data

    Attributes:

    """

    def __init__(self, cfg, data):
        [self.X, self.Y] = data
        self.transform = data_transform(cfg)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data, labels = self.X[idx].astype('float32'), self.Y[idx]
        if self.transform is not None:
            aug = self.transform(self.X[idx])  # conversion to float32 implied
            return data, aug, labels
        return data, labels


class HierEEGDataset(Dataset):
    """
    Dataset of single subject (epoched data)

    Args:
        data: numpy array of data

    Attributes:

    """

    def __init__(self, cfg, data, test=False, known=None):
        [self.X, self.Y] = data
        self.test = test
        self.transform = data_transform(cfg, test) # conversion to float32 implied
        self.known = known

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx_list):
        x, y = [], []
        for idx in idx_list:
            x.append(torch.from_numpy(self.X[idx]).float())
            y.append(torch.from_numpy(self.Y[idx]).long())
        return self.transform(torch.stack(x)), torch.stack(y)


class HierBatchSampler(Sampler):
    def __init__(self,
                 batch_size: int,
                 dataset: HierEEGDataset,
                 drop_last: bool = False) -> None:

        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.drop_last = drop_last
        self.labels = dataset.Y
        self.tree = self._create_labels_tree(dataset)

    def _create_labels_tree(self, dataset):
        tree = {}
        for i in range(len(dataset)):
            *super_labels, class_label = dataset.Y[i]
            current_tree = tree
            for label in super_labels:
                if label not in current_tree:
                    current_tree[label] = {}
                current_tree = current_tree[label]
            # Now we're at the class level, we initialize it as a list to contain indices
            if class_label not in current_tree:
                current_tree[class_label] = []
            current_tree[class_label].append(i)
        return tree

    @staticmethod
    def random_unvisited_sample(label, tree):
        current_tree = tree
        top_level = True
        while type(current_tree) is dict:
            if top_level:
                random_label = label
                if len(current_tree.keys()) != 1:
                    while random_label == label: # sample from label other than current one
                        random_label = random.choice(list(current_tree.keys()))
            else:
                random_label = random.choice(list(current_tree.keys()))
            current_tree = current_tree[random_label]
            top_level = False
        idx = random.sample(current_tree, 1)
        return idx[0]

    def _get_unvisited_sample(self, label, tree, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.random_unvisited_sample(label, tree)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        idx = remaining[torch.randint(len(remaining), (1,))]
        return idx

    def _get_subtree_from_path(self, path, tree):
        current_tree = tree
        for node in path:
            current_tree = current_tree[node]
        return current_tree

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()

        if len(self.dataset) % self.batch_size != 0:
            if self.drop_last:
                # remove tail of data to make it evenly divisible.
                indices = indices[:-(len(self.dataset) % self.batch_size)]
            else:
                # add extra samples to make it evenly divisible
                indices += indices[:(len(self.dataset) % self.batch_size)]

        remaining = list(set(indices).difference(visited))

        while len(remaining) > self.batch_size:
            anchor = indices[torch.randint(len(indices), (1,))]
            # anchor = random.choice(remaining)
            batch.append(anchor)
            visited.add(anchor)
            # e.g., anchor: c=6, H-SU. ( t = 1, p = 0, m = 1 -> hand, static pose, dynamic motion (Slide RLUD) )
            label_list = self.labels[anchor]
            for i, label in enumerate(label_list[::-1]):  # reverse order
                current_tree = self._get_subtree_from_path(label_list[:-(i+1)], self.tree)
                idx = self._get_unvisited_sample(label, current_tree, visited, indices, remaining)
                batch.append(idx)
                visited.add(idx)

            remaining = list(set(indices).difference(visited))

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
