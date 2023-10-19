"""
Load and preprocess NeuroSkin dataset
"""

from .dataload import *
from .preprocess import *

import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('error')

NORMALIZERS = {
    'zscore': Preprocessor(standard_scale, channel_wise=True),
    'ewma': Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000),
    'fixedscale': Preprocessor(fixed_scale, channel_wise=True),
    'none': lambda x: x
}

# Sensorimotor 22 channels
CHANS_22 = ['AF3', 'F1', 'F3', 'C1', 'C3', 'CP1', 'CP3', 'P1', 'P3',
            'AF4', 'F2', 'F4', 'C2', 'C4', 'CP2', 'CP4', 'P2', 'P4']
CHANS_18_square = ['F1', 'F3', 'C1', 'C3', 'C5', 'P1', 'P3', 'CP1', 'CP3',
                   'CP2', 'CP4', 'F2', 'F4', 'C2', 'C4', 'C6', 'P2', 'P4']
CHANS_18_rect = ['FC1', 'FC3', 'FC5', 'C1', 'C3', 'C5', 'CP5', 'CP1', 'CP3',
                 'CP2', 'CP4', 'FC2', 'FC4', 'FC6', 'C2', 'C4', 'C6', 'CP6']
CHANS_10 = ['C1', 'C3', 'C5', 'CP1', 'CP3',
            'CP2', 'CP4', 'C2', 'C4', 'C6']
CHANS_16 = ['F3', 'F7', 'FC5', 'FC1', 'C3', 'CP5', 'CP1', 'P3',
            'F4', 'F8', 'FC6', 'FC2', 'C4', 'CP6', 'CP2', 'P4']
CHANS_28 = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
            'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
CHANS_56 = ['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'C1',
            'C3', 'C5', 'T7', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1',
            'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FT10', 'FT8', 'FC6', 'FC4', 'FC2', 'C2', 'C4',
            'C6', 'T8', 'TP10', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO4', 'O2']

CHANS_DICT = {10:CHANS_10, 16:CHANS_16, '18R':CHANS_18_rect, '18S': CHANS_18_square, 22:CHANS_22, 28:CHANS_28, 56:CHANS_56}

CLASS_DICT = {'Arm/Slide/Right': ['Arm/Slide/R', 'A-SR'], 'Arm/Slide/Left': ['Arm/Slide/L','A-SL'],
              'Arm/Pinch': ['Arm/Pinch', 'A-Pin'], 'Arm/Spread': ['Arm/Spread', 'A-Spr'],
              'Palm/Slide/Right': ['Hand/Slide/R', 'H-SR'], 'Palm/Slide/Left': ['Hand/Slide/L', 'H-SL'],
              'Palm/Slide/Up': ['Hand/Slide/U', 'H-SU'], 'Palm/Slide/Down': ['Hand/Slide/D', 'H-SD'],
              'Palm/Pinch': ['Hand/Pinch', 'H-Pin'], 'Palm/Press': ['Hand/Tap', 'H-Tap'],
              'Palm': 'Hand', 'Press': 'Tap', 'Slide/Right': 'Slide'}

labels_dict = {
    'touch': {
        'Arm': [0, 1, 2, 3],
        'Hand': [4, 5, 6, 7, 8, 9]
    },
    'pose': {
        'A/Index': [0, 1],
        'A/ThumbInd': [2, 3],
        'H/Index': [4, 5, 6, 7, 9],
        'H/ThumbInd': [8]
    },
    'motion': {
        'A/Ind/S': [0, 1],
        'A/ThumbInd/P': [2, 3],
        'H/Ind/T': [9],
        'H/Ind/S': [4, 5, 6, 7],
        'H/ThumbInd/P': [8]
    },
    'direction': {
        'AS/Hori': [0, 1],
        'HS/Hori': [4, 5],
        'HS/Vert': [6, 7],
        'A/N': [2, 3],
        'H/N': [8, 9]
    },
}


def neuroskin_mne(cfg, subject=None, verbose=False):

    # Load epochs
    epochs = mne.read_epochs(cfg.path + f'{subject}-epo.fif', preload=True, verbose=verbose)
    new_keys = [k.replace('_', '/') for k, _ in epochs.event_id.items()]
    epochs.event_id = dict(zip(new_keys, list(epochs.event_id.values())))

    # Select and reorder channels
    if cfg.select_channels:
        epochs = epochs.reorder_channels(CHANS_DICT[cfg.select_channels])

    # Select classes
    if cfg.select_classes:
        # MNE Epochs object supports indexing by class name
        epochs_list = [epochs[c] for c in cfg.select_classes]
        epochs = mne.concatenate_epochs(epochs_list)

    # Merge classes
    if cfg.merge_classes:
        # Select (filter) classes first
        epochs = mne.concatenate_epochs([epochs[c] for c in cfg.merge_classes.values()])
        # Merge
        event_dict = {}
        for i, (name, sub_classes) in enumerate(cfg.merge_classes.items()):
            event_ids = []
            for sc in sub_classes:
                ev = [v for k, v in epochs.event_id.items() if sc in k]
                event_ids.extend(ev)
            epochs.events = mne.merge_events(epochs.events, event_ids, i)
            event_dict[i] = name
    else:
        event_dict = {v: k for k, v in epochs.event_id.items()}

    cc, Ya = np.unique(epochs.events[:, 2], return_inverse=True)

    Y, nclasses = create_multilabels(Ya, len(cc), labels_dict, cfg.select_labels)

    # Preprocess the data
    from braindecode.preprocessing.preprocess import Preprocessor  #preprocess,
    preprocessors = []
    if cfg.scaling:
        preprocessors.append(Preprocessor(Scaler(epochs, scale=1e6)))
    if cfg.normalization != 'none': # trial normalization, e.g., channel-wise z-score normalization
        preprocessors.append(NORMALIZERS[cfg.normalization])
    if cfg.alignment:
        preprocessors.append(Preprocessor(EuclideanAlignment(epochs)))
    for p in preprocessors:
        if verbose: print('Preprocessing...')
        p.apply(epochs)

    # Update configs
    c = [event_dict[c] for c in cc]
    class_long = [CLASS_DICT[name][0] if name in CLASS_DICT.keys() else name for name in c]
    class_abbr = [CLASS_DICT[name][1] if name in CLASS_DICT.keys() else name for name in c]
    class_set = ['Arm', 'Hand', 'Tap', 'Pinch', 'Spread', 'Slide', 'R', 'L', 'U', 'D']

    cfg.update({
        "n_classes": nclasses, "n_labels": Y.shape[-1],
        "n_chans": epochs.info['nchan'], "n_times": len(epochs.times),
        'ch_names': epochs.ch_names, "sfreq": epochs.info['sfreq'],
        "l_freq": epochs.info['highpass'], "h_freq": epochs.info['lowpass'],
        "tmin": epochs.tmin, "tmax": epochs.tmax, "tlen": epochs.tmax-epochs.tmin,
        "class_set": class_set, "class_abbr": class_abbr, "class_long": class_long
    })

    X = epochs.get_data()

    return X, Y
