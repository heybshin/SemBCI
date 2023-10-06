from copy import deepcopy

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR

from torchmetrics import ConfusionMatrix, Accuracy, AUROC
from pytorch_lightning import LightningModule
from hyptorch import ToPoincare

from losses import DisentLoss
from .modules import Encoder, MIEstimator, Classifier
from hierarchy.tree import get_distances_similarities
from utils import compute_osr_metrics, compute_hier_metrics


class SAClassifier(LightningModule):
    """
    Semantics-Aligned Learning Framework for EEG Classification
    """
    def __init__(self, cfg, criterion):
        super().__init__()
        self.disentangle = cfg.DISENT
        self.pretraining = False

        self.r_loss = cfg.LOSS.reg
        self.loss_name = cfg.LOSS.name
        self.loss_fn = criterion
        self.loss_cls = nn.CrossEntropyLoss()

        self.class_names = cfg.DATA.class_abbr
        self.cfg = cfg.TRAIN

        nclasses = cfg.DATA.n_classes.all
        ldim = cfg.MODEL.latent_dim
        self.encoder = Encoder(cfg)
        self.classifier = Classifier(ldim, ldim//2, nclasses)

        if self.disentangle:
            estimator = MIEstimator(ldim)
            self.loss_dis = DisentLoss(estimator, type=cfg.DISENT)

        if self.loss_name == 'dag':
            self.embed_layer = nn.Sequential(nn.Linear(ldim, cfg.MODEL.n_nodes), nn.Sigmoid())
            self.graph = criterion.graph

        if self.loss_name == 'hicon':
            self.hyp = ToPoincare(c=0.1, ball_dim=ldim, riemannian=True) if cfg.LOSS.hyp else None

        # Metrics
        self.acc = Accuracy(nclasses)
        self.roc = AUROC(task="multiclass", average="macro", num_classes=nclasses)
        self.distances, self.best_hiersim = get_distances_similarities(cfg)

        # Logging
        self.best_metrics = {'ACC': 0, 'AUROC': 0, 'hier_dist_mistake': 100}

        stages = ['train', 'val', 'test']
        self.best_cm = {}
        self.best_acc = {}
        for stage in stages:
            setattr(self, f'{stage}_cm', ConfusionMatrix(num_classes=nclasses))
            self.best_acc[stage] = 0.
            self.best_cm[stage] = None

        self.save_hyperparameters()

    def set_pretrain_mode(self, pretraining=True, weights=None):
        self.pretraining = pretraining

        if not pretraining:
            self.loss_name = 'ce'
            if weights is not None:
                self.load_from_checkpoint(weights)

        # Freeze/unfreeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = pretraining

        # Freeze/unfreeze the other parameters (opposite of the encoder)
        for name, param in self.named_parameters():
            if "encoder" not in name:
                param.requires_grad = not pretraining

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        z = self.forward(x)
        total_loss, y_hat = self.compute_total_loss(z, y, self.loss_name)
        acc = self.acc(y_hat, y[:, -1])

        self.log(f'L{self.loss_name}', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': total_loss, 'acc': acc, 'features': z, 'labels': y[:, -1], 'logits': y_hat}

    def predict_step(self, batch, batch_idx):
        return self.forward(batch[0])

    #TODO: get rid of epoch_end
    def _epoch_end(self, stage, acc):
        if acc > self.best_acc[stage]:
            self.best_acc[stage] = acc
            cm = getattr(self, f"{stage}_cm")
            self.best_cm[stage] = deepcopy(cm.compute().detach().cpu().numpy().astype(int))
            cm.reset()

    @staticmethod
    def process_batch(batch):
        if len(batch[0].shape) > 3: # if augmented
            batch = [b.squeeze(0) for b in batch]
        if len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch

        return x, y

    def compute_total_loss(self, z, y, loss_fn):
        total_loss = 0

        if self.disentangle:
            z, zi = z
            Ldis = self.loss_dis(z, zi)
            self.log('Ldis', Ldis, on_step=False, on_epoch=True, prog_bar=True)
            total_loss += 0.1 * Ldis

        y_hat = self.classifier(z)
        loss_cls = self.loss_cls(y_hat, y[:, -1])

        if loss_fn == 'ce':
            total_loss += loss_cls

        elif loss_fn == 'hice':
            loss = self.loss_fn(y_hat, y[:, -1])
            total_loss += self.r_loss * loss

        elif loss_fn == 'hicon':
            z = self.hyp(z) if self.hyp is not None else z
            loss = self.loss_fn(z, y, self.current_epoch)
            total_loss += self.r_loss * loss

        elif loss_fn == 'dag':
            z = self.embed_layer(z)
            loss, y_hat = self.loss_fn(z, y[:, -1])
            total_loss += self.r_loss * loss + loss_cls

        elif loss_fn == 'arpl':
            loss, y_hat = self.loss_fn(z, y[:, -1])
            total_loss += self.r_loss * loss

        return total_loss, y_hat

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.pretraining:
            optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
            return [optimizer], [scheduler]

        if self.cfg.optim == "adaptive":
            optimizer = optim.Adam(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        if self.cfg.optim == "cyclic":
            optimizer = optim.RMSprop(params, lr=self.cfg.lr)
            scheduler = CyclicLR(optimizer=optimizer,
                                 base_lr=self.cfg.lr,
                                 max_lr=1.5e-3,
                                 step_size_up=self.cfg.step_size,
                                 mode=self.cfg.mode,
                                 cycle_momentum=False,
                                 gamma=self.cfg.gamma)

        if self.cfg.optim == "cosine":
            optimizer = optim.SGD(params, lr=self.cfg.lr,
                                  momentum=self.cfg.momentum,
                                  weight_decay=self.cfg.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=(self.cfg.n_epochs / 5), eta_min=0)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class SAClosedSetClassifier(SAClassifier):
    """
    Semantics-Aligned Learning Framework for Closed-Set EEG Classification
    """
    def __init__(self, cfg, criterion):
        super(SAClosedSetClassifier, self).__init__(cfg, criterion)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, stage='val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, stage='test')

    def validation_epoch_end(self, outputs):
        labels = torch.cat([o['labels'] for o in outputs]).detach().cpu().numpy()
        preds = torch.cat([o['logits'] for o in outputs]).detach().cpu().numpy()
        acc = torch.stack([o['acc'] for o in outputs]).mean().detach().cpu().numpy()
        self._epoch_end('val', acc)

        # Hierarchical metrics
        hierarchy = 'graph' if self.loss_name == 'dag' else 'tree'
        hier_metrics = compute_hier_metrics(preds, labels,
                                            self.class_names,
                                            hierarchy=hierarchy,
                                            distances=self.distances,
                                            best_hiersim=self.best_hiersim)

        if hier_metrics['hier_dist_mistake'] < self.best_metrics['hier_dist_mistake']:
            self.best_metrics.update(hier_metrics)

        self.best_metrics['ACC'] = self.best_acc['val']

    def test_epoch_end(self, outputs):
        acc = torch.stack([o['acc'] for o in outputs]).mean().detach().cpu().numpy()
        self._epoch_end('test', acc)

    def _shared_eval(self, batch, stage='val'):
        x, y = self.process_batch(batch)
        z = self.forward(x)
        if self.disentangle:
            zr, zi = z
        else:
            zr = z
        if self.loss_name == 'arpl':
            loss, y_hat = self.loss_fn(zr, y[:, -1])
        elif self.loss_name == 'dag':
            zr = self.embed_layer(zr)
            loss, y_hat = self.loss_fn(zr, y[:, -1])
        else:
            y_hat = self.classifier(zr)
            loss = self.loss_cls(y_hat, y[:, -1])

        acc = self.acc(y_hat, y[:, -1])

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        getattr(self, f"{stage}_cm")(y_hat, y[:, -1])

        return {'loss': loss, 'acc': acc, 'features': z, 'labels': y[:,-1], 'logits': y_hat}


class SAOpenSetClassifier(SAClassifier):
    """
    Semantics-Aligned Learning Framework for Open-Set EEG Classification
    """
    def __init__(self, cfg, criterion):
        super(SAOpenSetClassifier, self).__init__(cfg, criterion)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = self.process_batch(batch)
        z = self.forward(x)
        if self.disentangle:
            zr, zi = z
        else:
            zr = z
        y_hat = self.classifier(zr)

        if dataloader_idx == 1:  # unknown dataloader
            return {'features': z, 'logits': y_hat}

        if self.loss_name == 'arpl':
            loss, y_hat = self.loss_fn(zr, y[:, -1])
        elif self.loss_name == 'dag':
            zr = self.embed_layer(zr)
            loss, y_hat = self.loss_fn(zr, y[:, -1])
        else:
            loss = self.loss_cls(y_hat, y[:, -1])

        acc = self.acc(y_hat, y[:, -1])

        self.log(f'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        getattr(self, f"val_cm")(y_hat, y[:, -1])

        return {'loss': loss, 'acc': acc, 'features': z, 'labels': y[:,-1], 'logits': y_hat}

    def validation_epoch_end(self, outputs):
        known, unknown = outputs
        pred_k = torch.cat([o['logits'] for o in known]).detach().cpu().numpy()
        pred_u = torch.cat([o['logits'] for o in unknown]).detach().cpu().numpy()
        y_k = torch.cat([o['labels'] for o in known]).detach().cpu().numpy()
        acc = torch.stack([o['acc'] for o in known]).mean().detach().cpu().numpy()
        self._epoch_end('val', acc)

        # Hierarchical metrics
        hierarchy = 'graph' if self.loss_name == 'dag' else 'tree'
        hier_metrics = compute_hier_metrics(pred_k, y_k,
                                            self.class_names,
                                            hierarchy=hierarchy,
                                            distances=self.distances,
                                            best_hiersim=self.best_hiersim)
        if hier_metrics['hier_dist_mistake'] < self.best_metrics['hier_dist_mistake']:
            self.best_metrics.update(hier_metrics)

        # Open-set recognition metrics
        osr_metrics = compute_osr_metrics(pred_k, pred_u)
        if osr_metrics['AUROC'] > self.best_metrics['AUROC']:
            self.best_metrics.update(osr_metrics)

        self.best_metrics['ACC'] = self.best_acc['val']