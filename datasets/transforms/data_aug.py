"""
Data augmentation transforms for EEG data.
"""

import torch
import numpy as np
from numbers import Real
from sklearn.utils import check_random_state
import torch.nn.functional as F


def identity(x):
    return x


def pick_channels_randomly(X, prob, rng):
    batch_size, n_channels, _ = X.shape
    unif_samples = torch.as_tensor(
        rng.uniform(0, 1, size=(batch_size, n_channels)),
        dtype=torch.float,
        device=X.device,
    )
    return torch.sigmoid(1000*(unif_samples - prob))


def make_permutation_matrix(X, mask, rng):
    batch_size, n_channels, _ = X.shape
    hard_mask = mask.round()
    batch_permutations = torch.empty(
        batch_size, n_channels, n_channels, device=X.device
    )
    for b, mask in enumerate(hard_mask):
        channels_to_shuffle = torch.arange(n_channels)
        channels_to_shuffle = channels_to_shuffle[mask.bool()]
        channels_permutation = np.arange(n_channels)
        channels_permutation[channels_to_shuffle] = rng.permutation(
            channels_to_shuffle
        )
        channels_permutation = torch.as_tensor(
            channels_permutation, dtype=torch.int64, device=X.device
        )
        batch_permutations[b, ...] = F.one_hot(channels_permutation)
    return batch_permutations


class ComposeTransforms:
    """Composes several transforms together.

    Parameters:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Transform1(),
        >>>     Transform2(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Transform(torch.nn.Module):
    def __init__(self, probability=1.0, random_state=None):
        super().__init__()
        assert isinstance(probability, Real), (
            f"probability should be a ``real``. Got {type(probability)}.")
        assert probability <= 1. and probability >= 0., \
            "probability should be between 0 and 1."
        self._probability = probability
        self.rng = check_random_state(random_state)

    def transform(self, X):
        """Transformation function to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the transform method.")

    def forward(self, X):
        """General forward pass for an augmentation transform."""
        X = torch.as_tensor(X).float()
        out_X = X.clone()

        # Samples a mask setting for each example whether they should stay
        # unchanged or not
        mask = self._get_mask(X.shape[0])
        num_valid = mask.sum().long()

        if num_valid > 0:
            out_X[mask, ...] = self.transform(out_X[mask, ...])

        return out_X

    def _get_mask(self, batch_size=None) -> torch.Tensor:
        """Samples whether to apply operation or not over the whole batch"""
        return torch.as_tensor(
            self.probability > self.rng.uniform(size=batch_size)
        )

    @property
    def probability(self):
        return self._probability


class GaussianNoise(Transform):
    """Randomly add white noise to all channels.
    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    std : float, optional
        Standard deviation to use for the additive noise. Defaults to 0.1.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.
    """

    def __init__(
        self,
        probability,
        std=0.1,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        self.std = std

    def transform(self, X):
        if isinstance(self.std, torch.Tensor):
            std = self.std.to(X.device)
        else:
            std = self.std
        noise = torch.from_numpy(
            self.rng.normal(
                loc=np.zeros(X.shape),
                scale=1
            ),
        ).float().to(X.device) * std
        transformed_X = X + noise
        return transformed_X


class ChannelsDropout(Transform):
    """Randomly set channels to flat signal.

    Parameters
    ----------
    probability: float
        Float setting the probability of applying the operation.
    proba_drop: float | None, optional
        Float between 0 and 1 setting the probability of dropping each channel.
        Defaults to 0.2.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument and to sample channels to erase. Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """
    def __init__(
        self,
        probability,
        p_drop=0.2,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )
        self.p_drop = p_drop

    def transform(self, X):
        mask = pick_channels_randomly(X, self.p_drop, self.rng)
        return X * mask.unsqueeze(-1)


class ChannelPermutation(Transform):
    """Randomly shuffle channels in EEG data matrix.

    Parameters
    ----------
    probability: float
        Float setting the probability of applying the operation.
    p_shuffle: float | None, optional
        Float between 0 and 1 setting the probability of including the channel
        in the set of permuted channels. Defaults to 0.2.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument, to sample which channels to shuffle and to carry the shuffle.
        Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """

    def __init__(
        self,
        probability,
        p_shuffle=0.2,
        ordered_ch_names=None,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )
        self.p_shuffle = p_shuffle
        self.ordered_ch_names = ordered_ch_names

        if ordered_ch_names is not None:
            self._init_permutation()

    def _init_permutation(self):
        assert (
            isinstance(self.ordered_ch_names, list) and
            all(isinstance(ch, str) for ch in self.ordered_ch_names)
        ), "ordered_ch_names should be a list of str."
        permutation = list()
        for idx, ch_name in enumerate(self.ordered_ch_names):
            new_position = idx
            d = ''.join(list(filter(str.isdigit, ch_name)))
            if len(d) > 0:
                d = int(d)
                if d % 2 == 0:  # pair/right electrodes
                    sym = d - 1
                else:  # odd/left electrodes
                    sym = d + 1
                new_channel = ch_name.replace(str(d), str(sym))
                if new_channel in self.ordered_ch_names:
                    new_position = self.ordered_ch_names.index(new_channel)
            permutation.append(new_position)
        self.permutation = torch.tensor(permutation, dtype=torch.int64)

    def transform(self, X):
        if self.p_shuffle == 0:
            return X
        mask = pick_channels_randomly(X, self.p_shuffle, self.rng) \
            if self.ordered_ch_names is None else self.permutation
        batch_permutations = make_permutation_matrix(X, mask, self.rng)
        return torch.matmul(batch_permutations, X)


class SmoothTimeMask(Transform):
    """Smoothly replace a randomly chosen contiguous part of all channels by
    zeros.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    mask_len_samples : int | torch.Tensor, optional
        Number of consecutive samples to zero out. Will be ignored if
        magnitude is not set to None. Defaults to 100.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    """

    def __init__(
        self,
        probability,
        mask_len_samples=100,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )

        assert (
            isinstance(mask_len_samples, (int, torch.Tensor)) and
            mask_len_samples > 0
        ), "mask_len_samples has to be a positive integer"
        self.mask_len_samples = mask_len_samples

    def transform(self, X):
        batch_size, n_channels, seq_len = X.shape

        mask_start = torch.as_tensor(self.rng.uniform(
            low=0, high=1, size=X.shape[0],
        )) * (seq_len - self.mask_len_samples)

        t = torch.arange(seq_len).float()
        t = t.repeat(batch_size, n_channels, 1)
        mask_start_per_sample = mask_start.view(-1, 1, 1)
        s = 1000 / seq_len
        mask = (torch.sigmoid(s * -(t - mask_start_per_sample)) +
                torch.sigmoid(s * (t - mask_start_per_sample - self.mask_len_samples))
                ).float().to(X.device)
        return X * mask
