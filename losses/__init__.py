import pickle
import torch
import torch.nn as nn

from hierarchy.tree import get_weighting
from .Hierarchical import HiCE, HiCon
from .DAG import RelationLoss
from .ARPL import ARPLoss
from .Disentangle import DisentLoss


def create_criterion(cfg, name=None):
    """
    Create a loss instance.
    """
    loss = cfg.LOSS.name if name is None else name

    if loss == "ce":
        if cfg.LOSS.smoothing:
            return nn.CrossEntropyLoss(reduction='mean', label_smoothing=cfg.LOSS.smoothing)
        return nn.CrossEntropyLoss()

    if loss == "hice":
        # Set hierarchy
        with open(cfg.LOSS.tree_path, "rb") as f:
            hierarchy = pickle.load(f)
        weights = get_weighting(hierarchy, alpha=cfg.LOSS.hxe_alpha)  # , normalize=cfg.hce_weight_norm)
        return HiCE(hierarchy, cfg.DATA.class_abbr, weights)

    if loss == 'dag':
        return RelationLoss(cfg)

    if loss == 'hicon':
        return HiCon(
            cfg.TRAIN.n_epochs, temperature=cfg.LOSS.hier_temp, alpha=cfg.LOSS.hier_alpha, penalty=cfg.LOSS.penalty
        )

    if loss == 'arpl':
        return ARPLoss(cfg.DATA.n_classes.all, cfg.MODEL.latent_dim, arpl_alpha=cfg.LOSS.arpl_alpha)

