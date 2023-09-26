import yaml
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar


class Config:
    def __init__(self, config_dict):
        self.update(config_dict)

    def update(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)


def load_yaml(dpath):
    # Loads a yaml file
    with open(dpath, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data, dpath):
    # Saves a yaml file
    with open(dpath, 'w') as f:
        yaml.dump(data, f)


def set_callbacks(cfg, path, subject, fold, pretrain_mode=False):
    # Set callbacks for training
    if cfg.TASK == 'csr':
        loss_monitor = 'val_loss'
        acc_monitor = 'val_acc'
    elif cfg.TASK == 'osr':
        loss_monitor = 'val_loss/dataloader_idx_0'
        acc_monitor = 'val_acc/dataloader_idx_0'

    dpath = path + f'sub{subject}'

    if pretrain_mode:
        fname = f'pretrain_{fold + 1}_' + '{epoch}-{train_loss:.2f}'
        ckpt = ModelCheckpoint(monitor='train_loss',
                               mode='min', save_top_k=1,
                               dirpath=save_path + f'sub{subject}',
                               save_on_train_epoch_end=True,
                               filename=fname)
        earlystop = EarlyStopping(monitor='Lhicon',
                                  mode='min', patience=20,
                                  check_on_train_epoch_end=True)
        return [ckpt, earlystop]

    fname_loss = f'fold_{fold + 1}_{{epoch}}-{{{loss_monitor}:.3f}}'
    fname_acc = f'fold_{fold + 1}_{{epoch}}-{{{acc_monitor}:.3f}}'

    ckpt_loss = ModelCheckpoint(monitor=loss_monitor,
                                mode='min', save_top_k=3,
                                dirpath=dpath, filename=fname_loss)
    ckpt_acc = ModelCheckpoint(monitor=acc_monitor,
                               mode='max', save_top_k=3,
                               dirpath=dpath, filename=fname_acc)

    callbacks = [ckpt_loss, ckpt_acc,
                 LearningRateMonitor(logging_interval='step'),
                 TQDMProgressBar(refresh_rate=10)]

    return callbacks


def compute_osr_metrics(pred_k, pred_u):
    x_u = np.max(pred_u, axis=1)
    if len(pred_k.shape) > 1:
        x_k = np.max(pred_k, axis=1)
    else:
        x_k = pred_k

    # Compute metrics for open-set recognition task
    tp, fp, tnr_at_tpr95 = get_curve_online(x_k, x_u)
    results = dict()
    # TNR
    results['TNR'] = 100. * tnr_at_tpr95

    # AUROC
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    results['AUROC'] = 100. * (-np.trapz(1. - fpr, tpr))

    # DTACC
    results['DTACC'] = 100. * (.5 * (tp / tp[0] + 1. - fp / fp[0]).max())

    return results


def get_curve_online(known, novel):
    known.sort()
    novel.sort()
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n

    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[l + 1:] = tp[l]
            fp[l + 1:] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[l + 1:] = np.arange(tp[l] - 1, -1, -1)
            fp[l + 1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l + 1] = tp[l]
                fp[l + 1] = fp[l] - 1
            else:
                k += 1
                tp[l + 1] = tp[l] - 1
                fp[l + 1] = fp[l]
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n

    return tp, fp, tnr_at_tpr95


def compute_hier_metrics(preds, labels, classes, distances=None, hierarchy='tree', best_hiersim=None):
    # Compute hierarchical metrics
    metrics = dict()
    if hierarchy == 'tree':
        # Convert logits to predicted class indices
        preds = np.argmax(preds, axis=1)

    # Compute top-1 error
    top1_err = np.mean(preds != labels)
    metrics['top1_err'] = top1_err

    # Compute average hierarchical distance
    hier_dists = np.array([distances[(classes[preds[i]], classes[labels[i]])] for i in range(len(labels))])
    metrics['avg_hier_dist'] = np.mean(hier_dists)

    # Compute hierarchical distance of mistakes
    mistakes = np.where(preds != labels)[0]
    assert np.all(mistakes == np.where(hier_dists != 0.)[0])
    metrics['hier_dist_mistake'] = np.sum(hier_dists[mistakes])

    # Compute average hierarchical precision
    hier_sim = 1 - hier_dists / distances.max_dist
    hier_prec = np.sum(hier_sim) / np.sum(best_hiersim)
    metrics['avg_hier_precision'] = np.mean(hier_prec)

    return metrics
