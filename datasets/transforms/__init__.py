from .data_aug import *
from .label_enc import *


def aug_time_mask(cfg):
    return SmoothTimeMask(probability=cfg.aug_prob, mask_len_samples=int(cfg.sfreq * 0.2))


def aug_chan_perm(cfg):
    return ChannelPermutation(probability=cfg.aug_prob)


def aug_chan_lat(cfg):
    return ChannelPermutation(probability=cfg.aug_prob, ordered_ch_names=cfg.ch_names)


def aug_chan_drop(cfg):
    return ChannelsDropout(probability=cfg.aug_prob, p_drop=0.5)


def aug_noise(cfg):
    return GaussianNoise(probability=cfg.aug_prob, std=0.1)


DATA_TRANSFORMS = {
    'time_mask': aug_time_mask,
    'ch_drop': aug_chan_drop,
    'ch_perm': aug_chan_perm,
    'ch_lat': aug_chan_lat,
    'noise': aug_noise
}


LABEL_TRANSFORMS = {
    'soft': SoftLabels,
    'one_hot': OneHot,
    'mlb': MLB
}


def data_transform(cfg, test=False):
    aug = cfg.data_aug
    if aug == 'none' or test:
        return identity
    if isinstance(aug, list):
        transforms = [DATA_TRANSFORMS[fn](cfg) for fn in aug]
        return ComposeTransforms(transforms)
    return DATA_TRANSFORMS[aug](cfg)


def label_transform(cfg):
    return LABEL_TRANSFORMS[cfg.label_enc](cfg) if cfg.label_enc != 'none' else None

