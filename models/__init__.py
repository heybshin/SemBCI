# from .DCN import *
# from .ViT import *
# from .SKMSNet import *
from .SAC import SAClosedSetClassifier, SAOpenSetClassifier

# Add more models here
MODELS = {
    # 'DCN': DeepConvNet2,
    # 'ViT': ViT,
    # 'SKMSNet': SKMSNet,
    'SAC': SAClosedSetClassifier,
    'SAO': SAOpenSetClassifier
}


def create_model(cfg, criterion, pretrained=False):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        the name of model
    """
    if cfg.TASK == 'csr':
        return SAClosedSetClassifier(cfg, criterion)
    if cfg.TASK == 'osr':
        return SAOpenSetClassifier(cfg, criterion)

    # if name not in MODELS.keys():
    #     raise KeyError("Unknown model:", name)
    # if pretrained:
    #     return MODELS[name].load_from_checkpoint(pretrained, cfg=cfg, criterion=criterion, strict=False)
    # return MODELS[name](cfg, criterion)


