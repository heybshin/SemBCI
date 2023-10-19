import os
import argparse
from datetime import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold

from configs import create_cfg
from datasets import create_dataset, create_datamodule
from models import create_model
from losses import create_criterion
from utils import set_callbacks


def main(args):

    # Set config
    cfg = create_cfg(args)  # add default configs to sweep configs

    # Set seed and cuda
    seed_everything(cfg.SEED, workers=True)

    # Set paths
    if cfg.TASK == 'osr':
        if len(cfg.DATA.unknown_classes) > 1:
            task = f'{cfg.TASK}/Set2'
        else:
            task = f'{cfg.TASK}/Set1'
    else:
        task = cfg.TASK

    now = datetime.now().isoformat(timespec='seconds').replace(":", "-").replace("/", "-")
    save_path = f'results/{task}/{cfg.LOSS.name}_dis{cfg.DISENT}_{cfg.SAMPLER}_sampler/{now}/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Subject-wise k-fold cross-validation
    for subject in range(cfg.DATA.n_subs):

        # Set dataset
        ds = create_dataset(cfg, subject)

        # Set criterion
        criterion = create_criterion(cfg)

        # Set k-fold
        kfold = StratifiedKFold(n_splits=cfg.NUM_FOLDS, shuffle=True)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(ds.data, ds.labels[:, -1])):
            print(f'SUBJECT {subject}, FOLD {fold + 1}')
            cfg.log_name = f'sub{subject}_fold{fold + 1}'

            # Set datamodule
            ds.train_ids, ds.val_ids = train_ids, val_ids
            dm = create_datamodule(cfg, ds)

            # Set model
            model = create_model(cfg, criterion)

            # Set logger and callbacks (use any logger you want)
            logger = CSVLogger("logs", name=cfg.log_name)

            # Pretrain if pretext task is given
            if cfg.LOSS == 'hicon':
                model.set_pretrain_mode(pretraining=True)
                pre_callbacks = set_callbacks(cfg, save_path, subject, fold, pretrain_mode=True)
                trainer = Trainer(max_epochs=cfg.TRAIN.n_epochs,
                                  limit_val_batches=0,
                                  logger=logger,
                                  callbacks=pre_callbacks,
                                  gradient_clip_val=0.5,
                                  accelerator='gpu',
                                  devices=args.gpus)

                trainer.fit(model, datamodule=dm)
                model.set_pretrain_mode(pretraining=False, weights=pre_callbacks[0].best_model_path)

            # Train
            callbacks = set_callbacks(cfg, save_path, subject, fold)
            trainer = Trainer(max_epochs=cfg.TRAIN.n_epochs,
                              accelerator='gpu',
                              devices=args.gpus,  # strategy='ddp',
                              callbacks=callbacks, logger=logger,
                              gradient_clip_val=0.5,
                              val_check_interval=0.5)

            trainer.fit(model, datamodule=dm)

            # END of fold loop
            torch.cuda.empty_cache()

        # END of subject loop


# --gpus 0  --kfold 5

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pretraining using offline dataset.')
    parser.add_argument('--epochs', '-e', help='Number of epochs per run.', type=int)
    parser.add_argument('--gpus', type=int, nargs="*", help="List of GPU indices")
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    try:
        main(args)
    except RuntimeError as e:
        print(e)
