"""MicroSTNet author-code evaluation protocol, isolated from all other FTQNet tasks.

Source of truth: https://github.com/pwner-web/MicroBioSTNet, lib/prepare.py
and main.py.  Input files remain unmodified at the root of dataset/.

Published implementation (not a leakage-free benchmark):
* Read OTU/taxon rows x original sample-order columns; do not collapse genera,
  rarefy, square-root, or perform dataset-dependent feature screening.
* Compute ONE mean and standard deviation over all taxa and samples (including
  the 30% validation period), Z-score globally and add 1 to BOTH x and y.
* Split time positions at floor(0.7*T), then build 12->H sliding windows
  separately inside train and validation; do not borrow train context for val.
* For the two-stream (non-LSTM) MicroSTNet author implementation, duplicate
  ~10% of validation windows into training, retaining them in validation too.
  This reproduces the public code, but validation is NOT an independent test.
* The ``test`` flag intentionally evaluates the exact same validation windows;
  its name only preserves FTQNet's pre-existing results/ file layout.

This is a protocol-level adaptation, not a reproduction of MicroSTNet's graph
model. Never claim these MAEs measure unseen independent test performance.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


MICROBIOME_FILES = {
    'M3-stool': 'M3_gut_abun.csv',
    'A-stool': 'A_Stool_abun.csv',
    'A-saliva': 'A_Saliva_abun.csv',
    'F4-stool': 'F4_gut_abun.csv',
}

# Defaults in the author's prepare.py/main.py; deliberately not new CLI flags.
PAPER_TRAIN_FRACTION = 0.7
PAPER_VALIDATION_REUSE_RATIO = 0.1
PAPER_ZSCORE_OFFSET = 1.0


@lru_cache(maxsize=6)
def _load_abundance(file_path):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            'Missing {}. Copy the unmodified CSV into dataset/ first.'.format(path)
        )
    table = pd.read_csv(path, encoding='utf-8-sig')
    if table.shape[0] == 0 or table.shape[1] < 3:
        raise ValueError('Expected taxon rows and sample columns in {}'.format(path))
    if table.iloc[:, 0].isna().any():
        raise ValueError('Taxon identifiers contain null entries: {}'.format(path))
    values = table.iloc[:, 1:].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Abundance matrix has negative, infinite or missing entries: {}'.format(path))

    # Author: X = X.astype(np.float32); means = X.mean(axis=(0, 2));
    # stds = X.std(axis=(0, 2)), where X has shape [taxon, 1, time].
    # Therefore this is a SINGLE scalar mean and std, not per taxon.
    mean = np.mean(values, dtype=np.float32)
    std = np.std(values, dtype=np.float32)
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Cannot globally Z-score a constant abundance matrix: {}'.format(path))
    normalized = ((values - mean) / std + PAPER_ZSCORE_OFFSET).T.copy()
    return normalized.astype(np.float32, copy=False), float(mean), float(std)


def _validate_path(root_path, data_path, dataset_name):
    if dataset_name not in MICROBIOME_FILES:
        raise ValueError('Unrecognized microbiome dataset: {}'.format(dataset_name))
    expected = MICROBIOME_FILES[dataset_name]
    if data_path != expected:
        raise ValueError('{} requires --data_path {}'.format(dataset_name, expected))
    return str((Path(root_path) / data_path).resolve())


def microbiome_feature_count(root_path, data_path, dataset_name):
    path = _validate_path(root_path, data_path, dataset_name)
    matrix, _, _ = _load_abundance(path)
    return matrix.shape[1]


def _author_reused_validation_indices(n_windows):
    """Exactly match prepare.py::cut_datasets for ratio=0.1.

    The public code first subsamples validation windows, then additionally
    includes the final validation window unless already selected.
    """
    count = int(PAPER_VALIDATION_REUSE_RATIO * n_windows)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    step = max(1, n_windows // count)
    indices = np.arange(0, n_windows, step, dtype=np.int64)[:count]
    if indices[-1] != n_windows - 1:
        indices = np.concatenate((indices, [n_windows - 1]))
    return indices


class Dataset_Microbiome(Dataset):
    """Return standard FTQNet five-item samples, with author's window split."""

    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='A_Stool_abun.csv', target='OT', scale=True,
                 timeenc=1, freq='d', cycle=1, dataset_name='A-stool'):
        if flag not in ('train', 'val', 'test'):
            raise ValueError('Only train, val and author-equivalent test are supported.')
        if features != 'M':
            raise ValueError('Microbiome is multivariate-to-multivariate: --features M.')
        if not scale:
            raise ValueError('Paper protocol always uses globally Z-scored data.')
        if size is None or len(size) != 3:
            raise ValueError('Expected size [seq_len, label_len, pred_len].')
        self.seq_len, self.label_len, self.pred_len = map(int, size)
        if self.seq_len <= 0 or self.pred_len <= 0 or not 0 <= self.label_len <= self.seq_len:
            raise ValueError('Invalid sequence or prediction length.')
        if cycle is None or int(cycle) < 1:
            raise ValueError('cycle must be a positive integer.')
        self.cycle = int(cycle)
        self.flag = flag
        self.dataset_name = dataset_name
        path = _validate_path(root_path, data_path, dataset_name)
        self.matrix, self.global_mean, self.global_std = _load_abundance(path)
        total = self.matrix.shape[0]
        self.n_channels = self.matrix.shape[1]
        split = int(total * PAPER_TRAIN_FRACTION)
        if split < self.seq_len + self.pred_len or total - split < self.seq_len + self.pred_len:
            raise ValueError('Not enough observations in each 70/30 interval for seq_len={}, pred_len={}'.format(
                self.seq_len, self.pred_len))

        # Index by GLOBAL time position.  Sliding windows never straddle 70%.
        train_origins = np.arange(0, split - self.seq_len - self.pred_len + 1, dtype=np.int64)
        val_origins = np.arange(split, total - self.seq_len - self.pred_len + 1, dtype=np.int64)
        if flag == 'train':
            # The author's main (non-LSTM) model duplicates part of validation
            # into training while keeping the entire validation set for scoring.
            reused = _author_reused_validation_indices(len(val_origins))
            self.origins = np.concatenate((train_origins, val_origins[reused]))
        else:
            self.origins = val_origins

        self.data_stamp = None  # FTQNet consumes no timestamps.
        if flag == 'train':
            print('{}: {} samples, {} original taxa, split 70/30 at {}'.format(
                dataset_name, total, self.n_channels, split))
            print('Paper author-code: global Z-score (full matrix), +1 offset, no feature filtering')
            print('train windows: {}; validation windows: {}; validation windows reused in train: {}'.format(
                len(train_origins), len(val_origins), len(self.origins) - len(train_origins)))
            print('WARNING: author protocol reuses validation data; reported MAE is NOT held-out test MAE.')
        if flag == 'test':
            print('Paper author protocol: test() reports the SAME 30% validation windows used for checkpoint selection.')

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        i = int(self.origins[index])
        s_end = i + self.seq_len
        y_begin = s_end - self.label_len
        y_end = s_end + self.pred_len
        x = self.matrix[i:s_end]
        y = self.matrix[y_begin:y_end]
        # The model does not read *_mark; produce identically shaped placeholders.
        x_mark = np.zeros((len(x), 4), dtype=np.float32)
        y_mark = np.zeros((len(y), 4), dtype=np.float32)
        cycle_index = torch.tensor(s_end % self.cycle, dtype=torch.long)
        return x, y, x_mark, y_mark, cycle_index

    def inverse_transform(self, data):
        return (np.asarray(data) - PAPER_ZSCORE_OFFSET) * self.global_std + self.global_mean


def microbiome_data_provider(args, flag):
    if flag == 'pred':
        raise ValueError('Paper protocol uses train and validation, not future unknown predictions.')
    dataset = Dataset_Microbiome(
        root_path=args.root_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, data_path=args.data_path,
        target=args.target, timeenc=0 if args.embed != 'timeF' else 1,
        freq=args.freq, cycle=args.cycle, dataset_name=args.data,
    )
    if dataset.n_channels != args.enc_in:
        raise ValueError('Model enc_in={} != number of original taxa {}'.format(
            args.enc_in, dataset.n_channels))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(flag == 'train'), drop_last=False,
                        num_workers=args.num_workers)
    print(flag, len(dataset))
    return dataset, loader
