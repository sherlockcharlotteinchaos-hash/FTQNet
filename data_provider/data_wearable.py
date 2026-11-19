"""Isolated MixBench-TS Wearable adapter for FTQNet.

Faithfully follows the *released MixBench loader*, including its 19-channel
selection, group-wise ffill/bfill, row-wise (not time-aware) sliding windows,
seed-42 subject split and all-subject StandardScaler. Deliberately does not
change the published loader's preprocessing decisions.

The fifth element is the original MixBench CycleNet wrapper's pooled-window
phase (window_index % W), adapted to FTQNet's required five-tuple.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

MIXBENCH_DROP_COLS = (
    'steps', 'distance', 'calories', 'missingness_score',
    'deviceId', 'ts_start', 'ts_end', 'lf', 'hf',
)
MIXBENCH_SUBJECT_SEED = 42
MIXBENCH_TRAIN_RATIO = 0.70
MIXBENCH_VAL_RATIO = 0.15


def mixbench_prepare_frame(path):
    """Implement MixBench Dataset_Wearable.__read_data__ verbatim in intent.

    Returns data AFTER author-provided channel dropping, per-subject sorting,
    within-subject forward/backward fill and final dropna. No resampling and no
    artificial timestamps are performed (as in the original code).
    """
    frame = pd.read_csv(path)
    for required in ('deviceId', 'ts_start', 'HR'):
        if required not in frame.columns:
            raise ValueError('Missing required column: ' + required)
    frame = frame.drop(columns=[
        c for c in MIXBENCH_DROP_COLS
        if c in frame.columns and c not in ('deviceId', 'ts_start')
    ])
    frame = frame.sort_values(['deviceId', 'ts_start']).reset_index(drop=True)
    frame = frame.groupby('deviceId', group_keys=False).apply(
        lambda group: group.ffill().bfill()
    )
    frame = frame.dropna().reset_index(drop=True)
    return frame


def wearable_feature_names(path, target='HR'):
    """Get actual feature order without loading the full dataset."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    for required in ('deviceId', 'ts_start', target):
        if required not in columns:
            raise ValueError('Wearable CSV missing required column: ' + required)
    kept = [c for c in columns if c not in MIXBENCH_DROP_COLS]
    features = [c for c in kept if c not in ('deviceId', 'ts_start') and c != target]
    return features + [target]


def mixbench_subject_splits(lengths):
    """Same sort-by-length + RandomState(42) shuffle as MixBench."""
    subject_ids = sorted(lengths, key=lengths.get, reverse=True)
    rng = np.random.RandomState(MIXBENCH_SUBJECT_SEED)
    rng.shuffle(subject_ids)
    count = len(subject_ids)
    train_n = max(1, int(count * MIXBENCH_TRAIN_RATIO))
    val_n = max(1, int(count * MIXBENCH_VAL_RATIO))
    return {
        'train': set(subject_ids[:train_n]),
        'val': set(subject_ids[train_n:train_n + val_n]),
        'test': set(subject_ids[train_n + val_n:]),
    }


class Dataset_Wearable_FTQNet(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 target='HR', scale=True, data_path='wearable_mixbench19.csv',
                 cycle=4):
        if flag not in ('train', 'val', 'test'):
            raise ValueError('Wearable uses train/val/test.')
        if features != 'M':
            raise ValueError('MixBench Wearable FTQNet runs require features=M.')
        if not size or len(size) != 3:
            raise ValueError('size must be [seq_len, label_len, pred_len].')
        self.seq_len, self.label_len, self.pred_len = map(int, size)
        if self.seq_len <= 0 or self.pred_len <= 0 or self.label_len < 0:
            raise ValueError('Invalid window lengths.')
        if cycle < 1:
            raise ValueError('Cycle must be positive.')
        self.flag = flag
        self.cycle = int(cycle)
        self.features = features
        self.target = target
        self.scale = scale
        self.path = Path(root_path) / data_path
        self.valid_indices = []
        self.subject_windows = {}
        frame = mixbench_prepare_frame(self.path)
        self.columns = [c for c in frame.columns
                        if c not in ('deviceId', 'ts_start') and c != target] + [target]
        self.subject_lengths = frame.groupby('deviceId').size().to_dict()
        self.subject_splits = mixbench_subject_splits(self.subject_lengths)
        split_ids = self.subject_splits[flag]

        required = self.seq_len + self.pred_len
        all_parts = []
        global_ptr = 0
        for subject_id, group in frame.groupby('deviceId', sort=False):
            values = group[self.columns].values
            n = len(values)
            all_parts.append(values)
            if subject_id in split_ids and n >= required:
                positions = list(range(global_ptr, global_ptr + n - required + 1))
                self.valid_indices.extend(positions)
                self.subject_windows[str(subject_id)] = len(positions)
            global_ptr += n

        if not all_parts:
            raise ValueError('Wearable input contains no eligible subjects.')
        data_raw = np.concatenate(all_parts, axis=0)
        # Intentional MixBench behavior: fit on *all* subjects, not only train.
        self.scaler = StandardScaler().fit(data_raw) if scale else None
        self.data_x = self.scaler.transform(data_raw) if scale else data_raw
        self.data_y = self.data_x
        if not np.isfinite(self.data_x).all():
            raise ValueError('Non-finite values after MixBench preprocessing.')
        if not self.valid_indices:
            raise ValueError(
                f'No valid {flag} windows for L={self.seq_len}, H={self.pred_len}.'
            )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.valid_indices):
            raise IndexError(index)
        start = self.valid_indices[index]
        end = start + self.seq_len
        y_begin = end - self.label_len
        y_end = end + self.pred_len
        # The time marks match MixBench's Dataset_Wearable: zero arrays.
        x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        y_mark = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)
        # Original MixBench data_factory's _CycleIndexDataset computes i % W
        # when wrapping Dataset_Wearable (which has no abs_starts or border1).
        cycle_index = torch.tensor(index % self.cycle, dtype=torch.long)
        return (self.data_x[start:end], self.data_y[y_begin:y_end],
                x_mark, y_mark, cycle_index)

    def manifest(self):
        return {
            'dataset': 'Wearable',
            'source': 'MixBench-TS released Dataset_Wearable implementation',
            'data_path': str(self.path.resolve()),
            'data_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
            'flag': self.flag,
            'channels': self.columns,
            'num_channels': len(self.columns),
            'num_subjects': len(self.subject_splits[self.flag]),
            'subjects_with_windows': len(self.subject_windows),
            'subject_ids': sorted(map(str, self.subject_splits[self.flag])),
            'windows': len(self),
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'stride': 1,
            'cycle': self.cycle,
            'cycle_source': 'MixBench CycleNet pooled window ordinal modulo W',
            'standardization': 'StandardScaler fitted on all subjects (original code)',
            'missing_data': 'per-subject ffill then bfill then dropna; no resampling',
        }


def wearable_data_provider(args, flag):
    ds = Dataset_Wearable_FTQNet(
        root_path=args.root_path, data_path=args.data_path,
        flag=flag, size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, cycle=args.cycle,
        scale=True,
    )
    if ds.data_x.shape[1] != args.enc_in:
        raise ValueError(f'FTQNet enc_in={args.enc_in} but Wearable has '
                         f'{ds.data_x.shape[1]} channels.')
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=(flag == 'train'), num_workers=args.num_workers,
                        drop_last=False)
    print(f'Wearable {flag}: {len(ds)} windows, '
          f'{len(ds.subject_windows)} eligible subjects, '
          f'{len(ds.columns)} channels (MixBench implementation).')
    return ds, loader
