"""ILI/Covid-19 adapters; isolated from every legacy Dataset_* implementation.

Reference: decisionintelligence/DUET at the commit in reference/DUET_COMMIT.txt.
TFB uses floor(N*.8), then floor(train_validation*.875), NOT floor(N*.7).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

HEALTH_NAMES = ('ILI', 'Covid-19')
SCHEMAS = {'ILI': (966, 7, 7), 'Covid-19': (1392, 948, 1)}


def read_health_csv(path):
    """Return dates and numeric wide data; accept official TFB long CSV too.

    Preserve first-seen channel order. Do not fill, interpolate, aggregate,
    sort irregular rows, discard columns, or fabricate a date index.
    """
    path = Path(path)
    # Detect duplicate wide headers before pandas can silently rename them.
    import csv
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV column names are not allowed.')
    raw = pd.read_csv(path)
    if 'date' not in raw:
        raise ValueError('Expected a date column in the official preprocessed CSV.')
    if {'cols', 'data'}.issubset(raw.columns):
        if set(raw.columns) != {'date', 'data', 'cols'}:
            raise ValueError('TFB long CSV must contain exactly date,data,cols.')
        if raw['cols'].isna().any():
            raise ValueError('Missing channel names in cols.')
        raw['date'] = pd.to_datetime(raw['date'], errors='raise')
        if raw.duplicated(['date', 'cols']).any():
            raise ValueError('Duplicate (date,cols) entries; aggregation is forbidden.')
        channel_order = raw['cols'].drop_duplicates().tolist()
        date_order = pd.DatetimeIndex(raw.loc[raw['cols'] == channel_order[0], 'date'])
        # Every channel must have the same ordered, complete date index.
        for _, block in raw.groupby('cols', sort=False):
            if not pd.DatetimeIndex(block['date']).equals(date_order):
                raise ValueError('TFB channels must have identical ordered timestamps.')
        wide = raw.pivot(index='date', columns='cols', values='data')
        wide = wide.reindex(index=date_order, columns=channel_order)
        dates = date_order
        values = wide
    else:
        if 'cols' in raw or 'data' in raw:
            raise ValueError('Ambiguous CSV: incomplete TFB long format.')
        dates = pd.DatetimeIndex(pd.to_datetime(raw['date'], errors='raise'))
        values = raw.drop(columns='date')
    if len(dates) == 0 or dates.hasnans or dates.has_duplicates:
        raise ValueError('Dates must be nonempty, unique and nonmissing.')
    if not dates.is_monotonic_increasing:
        raise ValueError('Dates must already be in ascending chronological order.')
    if values.shape[1] == 0:
        raise ValueError('No feature columns found.')
    values = values.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError('NaN/Inf/missing channel-date entries found; no automatic filling.')
    # Consistent string names in manifests and S/MS column selection.
    values.columns = values.columns.map(str)
    return dates, values


def tfb_borders(n):
    tv_end = int(n * 0.8)
    train_end = int(tv_end * 0.875)
    return train_end, tv_end, n


class Dataset_Health(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='ILI.csv', target='OT', scale=True, timeenc=1,
                 freq='w', cycle=52, dataset_name='ILI', strict_schema=True):
        if flag not in ('train', 'val', 'test'):
            raise ValueError('Health datasets support train/val/test, not Dataset_Pred.')
        if dataset_name not in HEALTH_NAMES:
            raise ValueError('Unknown health dataset: ' + dataset_name)
        if size is None or len(size) != 3:
            raise ValueError('size must be [seq_len,label_len,pred_len].')
        self.seq_len, self.label_len, self.pred_len = map(int, size)
        if self.seq_len <= 0 or self.pred_len <= 0 or not 0 <= self.label_len <= self.seq_len:
            raise ValueError('Invalid sequence, label or forecast length.')
        if cycle is None or cycle <= 0:
            raise ValueError('cycle must be positive.')
        self.cycle = int(cycle)
        self.flag = flag
        self.dataset_name = dataset_name
        self.path = Path(root_path) / data_path
        dates, frame = read_health_csv(self.path)
        n_rows, n_channels, days = SCHEMAS[dataset_name]
        if strict_schema and frame.shape != (n_rows, n_channels):
            raise ValueError('{} requires {} timestamps x {} channels; got {}. '
                             'Use the DUET release, not a similarly named dataset. '
                             '--health_strict_schema 0 is for nonbenchmark/debug data only.'
                             .format(dataset_name, n_rows, n_channels, frame.shape))
        if len(dates) > 1:
            deltas = np.diff(dates.asi8)
            if not np.all(deltas == pd.Timedelta(days=days).value):
                raise ValueError('{} must have regular {}-day timestamps.'.format(dataset_name, days))
        if features in ('S', 'MS'):
            if target not in frame:
                raise ValueError('S/MS requires an existing --target column, got ' + target)
            if features == 'S':
                frame = frame[[target]]
            else:
                frame = frame[[c for c in frame if c != target] + [target]]
        elif features != 'M':
            raise ValueError('features must be M, MS or S.')
        self.columns = list(frame.columns)
        self.split_ends = tfb_borders(len(frame))
        train_end, tv_end, n = self.split_ends
        borders = {'train': (0, train_end),
                   'val': (train_end - self.seq_len, tv_end),
                   'test': (tv_end - self.seq_len, n)}
        self.border1, self.border2 = borders[flag]
        if train_end < self.seq_len or self.border1 < 0:
            raise ValueError('Training history is too short for seq_len.')
        if self.border2 - self.border1 < self.seq_len + self.pred_len:
            raise ValueError('{} split is too short for seq_len/pred_len.'.format(flag))
        all_values = frame.to_numpy(dtype=np.float64)
        self.scaler = StandardScaler().fit(all_values[:train_end])
        self.scale = scale
        data = self.scaler.transform(all_values) if scale else all_values
        self.data_x = data[self.border1:self.border2].astype(np.float32)
        self.data_y = self.data_x
        # FTQNet does not consume marks. Native calendar marks avoid any new
        # dependency on the legacy time_features implementation for weekly data.
        stamp = np.stack([dates.month, dates.day, dates.dayofweek], axis=1)
        self.data_stamp = stamp[self.border1:self.border2].astype(np.float32)
        self.cycle_index = np.arange(self.border1, self.border2) % self.cycle

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        s_end = index + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        return (self.data_x[index:s_end], self.data_y[r_begin:r_end],
                self.data_stamp[index:s_end], self.data_stamp[r_begin:r_end],
                torch.tensor(self.cycle_index[s_end], dtype=torch.long))

    def inverse_transform(self, data):
        array = np.asarray(data)
        return self.scaler.inverse_transform(array.reshape(-1, array.shape[-1])).reshape(array.shape)

    def manifest(self):
        return {'dataset': self.dataset_name, 'data_path': str(self.path.resolve()),
                'data_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'columns': self.columns, 'split_ends': list(self.split_ends),
                'split': self.flag, 'slice': [self.border1, self.border2],
                'windows': len(self), 'scaler_fit_rows': self.split_ends[0],
                'normalized_metrics': bool(self.scale)}


class Dataset_ILI(Dataset_Health):
    def __init__(self, *args, **kwargs):
        kwargs['dataset_name'] = 'ILI'
        super().__init__(*args, **kwargs)


class Dataset_Covid19(Dataset_Health):
    def __init__(self, *args, **kwargs):
        kwargs['dataset_name'] = 'Covid-19'
        super().__init__(*args, **kwargs)


def health_data_provider(args, flag):
    if flag == 'pred':
        raise ValueError('Use train/val/test for DUET rolling health benchmarks.')
    dataset_type = {'ILI': Dataset_ILI, 'Covid-19': Dataset_Covid19}[args.data]
    dataset = dataset_type(
        root_path=args.root_path, data_path=args.data_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
        target=args.target, cycle=args.cycle, freq=args.freq,
        strict_schema=bool(getattr(args, 'health_strict_schema', 1)))
    if dataset.data_x.shape[1] != args.enc_in:
        raise ValueError('enc_in={} but loaded {} channels.'.format(args.enc_in, dataset.data_x.shape[1]))
    # Actual DUET implementation: train=True, validation/test=False.
    drop_last = flag == 'train' and bool(getattr(args, 'health_train_drop_last', 1))
    # Match DUET's shuffled validation batch order and mean-of-batch losses.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=flag != 'test',
                        num_workers=args.num_workers, drop_last=drop_last)
    if len(loader) == 0:
        raise ValueError('No batches: reduce batch_size or set --health_train_drop_last 0.')
    print('{} {}: {} windows, {} batches, drop_last={}'.format(args.data, flag, len(dataset), len(loader), drop_last))
    return dataset, loader


COVID_GROUPS = ("New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths")


def covid_group_indices(columns: Sequence[str], strict: bool = False) -> Dict[str, np.ndarray]:
    """Identify metric by exact final semicolon-delimited TFB column token."""
    result = {group: [] for group in COVID_GROUPS}
    for index, col in enumerate(columns):
        token = str(col).rsplit(";", 1)[-1].strip()
        if token not in result:
            raise ValueError(f"Unrecognized COVID-19 metric type: {col!r}")
        result[token].append(index)
    if strict and (len(columns) != 948 or any(len(v) != 237 for v in result.values())):
        raise ValueError("Official COVID-19 must have 237 channels per each of four metric types.")
    return {key: np.asarray(val, dtype=np.int64) for key, val in result.items()}


def covid_training_zero_floor(file: str | Path, expected_columns: Sequence[str]) -> tuple[np.ndarray, str]:
    """Calculate normalized raw zero from *training data only*.

    DO NOT simply clip normalized predictions to 0: after StandardScaler, a
    raw COVID-19 count of zero is generally a negative normalized value.
    """
    file = Path(file)
    _, frame = read_health_csv(file)
    expected = list(map(str, expected_columns))
    if list(frame.columns) != expected:
        raise ValueError("The channel order changed relative to the saved model")
    if frame.shape != (1392, 948):
        raise ValueError("This dedicated calibration only supports official COVID-19")
    covid_group_indices(expected, strict=True)
    train_end, _, _ = tfb_borders(len(frame))
    train = frame.iloc[:train_end].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(train)
    normalized_zero = scaler.transform(np.zeros((1, len(expected)), dtype=np.float64))[0]
    if not np.isfinite(normalized_zero).all():
        raise ValueError("Training-only normalized zero is not finite")
    return normalized_zero, hashlib.sha256(file.read_bytes()).hexdigest()
