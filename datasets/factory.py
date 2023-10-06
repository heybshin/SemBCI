import tqdm
from .neuroskin import *
from .dataload import EEGDataModule, OSRDataModule, DataContainer, extract_unknown


# Add more datasets here
DATASETS = {
    # 'neuroskinMAT': neuroskin_mat,
    'neuroskinMNE': neuroskin_mne,
}


def create_dataset(cfg, subject=None):
    dname, dtype = cfg.DATA.name, cfg.DATA.type

    # If subject-independent setting
    if subject is None:
        all_data = {}
        for s in tqdm.tqdm(range(cfg.DATA.n_subs), desc='Loading Data', colour='green'):
            all_data[s] = DATASETS[dname + dtype](cfg.DATA, s)
        return all_data

    data, labels = DATASETS[dname + dtype](cfg.DATA, subject, verbose=True)

    # Create data container with unknown set
    if cfg.TASK == 'csr':
        unknown_set = None
    if cfg.TASK == 'osr':
        (data, labels), unknown_set = extract_unknown(cfg.DATA, data, labels)

    return DataContainer(data=data, labels=labels, unknown_set=unknown_set)


def create_datamodule(cfg, dataset: DataContainer):
    if cfg.TASK == 'csr':
        return EEGDataModule(cfg, dataset)
    if cfg.TASK == 'osr':
        return OSRDataModule(cfg, dataset)
