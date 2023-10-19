"""
Helper functions for preprocessing EEG data
"""

import numpy as np
import pandas as pd
import mne
from scipy.linalg import sqrtm, inv
from braindecode.preprocessing.preprocess import Preprocessor, exponential_moving_standardize
from sklearn.preprocessing import scale as standard_scale

def create_multilabels(Y, n_classes, labels_dict, super_labels):
    # Map classes to hierachical multilabel
    class_mappings = np.full((n_classes, len(super_labels)), -1)  # initialize matrix with -1, to indicate no assignment yet
    ncls_dict = {'all': n_classes}
    for i, super_label in enumerate(super_labels):
        sub_labels = labels_dict[super_label]
        ncls_dict[super_label] = len(sub_labels)
        for j, (_, classes) in enumerate(sub_labels.items()):
            for c in classes:
                class_mappings[c, i] = j
    Y = np.c_[class_mappings[Y], Y] # associate each sample with multi-labels and append original class labels to last column

    return Y, ncls_dict


def zscore(data: np.ndarray, axis=-1):
    return (data - data.mean(axis, keepdims=True)) / (data.std(axis, keepdims=True) + 1e-12)


def fixed_scale(data: np.ndarray):
    return data / np.max(np.abs(data))


def ewma(data: np.ndarray, alpha=0.999):
    return exp_moving_whiten(data, factor_new=1 - alpha)


def exp_moving_whiten(data: np.ndarray, factor_new=0.001, init_block_size=1000, eps=1e-4):
    """
    Perform exponential running standardization.
    https://github.com/robintibor/braindecode/blob/master/braindecode/datautil/signalproc.py
    """
    if data.shape[0] == 1:
        data = data.squeeze(0)
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return np.expand_dims(standardized.T, axis=0)


class Preproc(object):

    def __init__(self, epochs: mne.Epochs):
        self.epochs = epochs

    def __call__(self, x: np.ndarray):
        raise NotImplementedError()


class Scaler(Preproc):
    """
    Preprocessor for performing unit conversion from volts to microvolts
    """

    def __init__(self, epochs, scale=1e6):
        super().__init__(epochs)
        self.scale = scale

    def __call__(self, x: np.ndarray):
        return (self.scale * x).astype(np.float32)


class EuclideanAlignment(Preproc):
    """
    Preprocessor for performing Euclidean Alignment (EA)
    https://github.com/SPOClab-ca/ThinkerInvariance/blob/784a8b8407c0b0c63557a032ab6b0520a7519fe9/dataload.py#L30
    """

    def __init__(self, epochs: mne.Epochs = None, data=None):
        if data is not None:
            x = data
        elif epochs is not None:
            x = epochs.get_data()
        else:
            raise ValueError('No data/epoched data specified!')
        r = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
        self.r_op = inv(sqrtm(r))
        if np.iscomplexobj(self.r_op):
            print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
                  "not, check data!!")
            self.r_op = np.real(self.r_op).astype(np.float32)
        elif not np.any(np.isfinite(self.r_op)):
            print("WARNING! Not finite values in R Matrix")

    def __call__(self, x: np.ndarray):
        return np.matmul(self.r_op, x)
