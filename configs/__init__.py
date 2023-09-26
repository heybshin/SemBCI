from utils import load_yaml, Config


def create_cfg(args):
    cfg = Config(load_yaml(f'./configs/experiment.yaml'))
    data_cfg = load_yaml(f'configs/{cfg.DATA}.yaml')
    model_cfg = load_yaml(f'configs/{cfg.LOSS}.yaml')

    cfg.update(data_cfg)
    cfg.update(model_cfg)

    if args.kfold:
        cfg.TRAIN.kfold = args.kfold
    if args.epochs:
        cfg.TRAIN.n_epochs = args.epochs
    if args.seed:
        cfg.SEED = args.seed

    return cfg
