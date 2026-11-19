"""EpiCastBench adapters for all five weekly epidemic datasets.

Read original CSVs in dataset/ without extracting, renaming or writing data.
Dataset-specific transformations and fixed-origin splits retain the uploaded implementation.
"""
import csv
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


FOUR_SPECS = {
    'Chickenpox-Hungary': ('hungary_chickenpox.csv', 522, 20,
                          '2005-01-03', '2014-12-29', None),
    'Dengue-Colombia': ('Dengue Colombia.csv', 835, 33,
                        '2006-12-31', '2022-12-25', 'COLOMBIA'),
    'Influenza-US-2': ('Influenza US 2.csv', 1200, 10,
                       '1997-09-28', '2020-09-20', 'HHS Regions'),
    'Dengue-Argentina': ('Dengue Argentina.csv', 92, 23,
                         '2019-09-22', '2021-06-20', 'ARGENTINA'),
}


def _numeric(series):
    result = pd.to_numeric(series, errors='raise')
    values = result.to_numpy(dtype=np.float64)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError('Incidence must be nonnegative and finite where observed.')
    return result


def mmwr_start(year, week):
    year, week = int(year), int(week)
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    first = jan4 - pd.Timedelta(days=(jan4.dayofweek + 1) % 7)
    jan4_next = pd.Timestamp(year=year + 1, month=1, day=4)
    next_first = jan4_next - pd.Timedelta(days=(jan4_next.dayofweek + 1) % 7)
    date = first + pd.Timedelta(weeks=week - 1)
    if week < 1 or date >= next_first:
        raise ValueError('Invalid MMWR year/week: {}/{}'.format(year, week))
    return date


@lru_cache(maxsize=4)
def _read_cached(path, name, size, mtime_ns):
    filename, length, channels, first, last, country = FOUR_SPECS[name]
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        skiprows = int(len(header) == 1 and 'INFLUENZA' in header[0].upper())
        if skiprows:
            header = next(reader, [])
        header = [str(c).strip() for c in header]
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV column names.')
    raw = pd.read_csv(path, skiprows=skiprows, encoding='utf-8-sig', low_memory=False)
    raw.columns = header
    dates = pd.date_range(first, last, freq='7D')
    coverage_matrix = None
    if country is None:
        if 'Date' not in raw:
            raise ValueError('Chickenpox requires the original Date column.')
        index = pd.DatetimeIndex(pd.to_datetime(raw.pop('Date'), format='%d/%m/%Y', errors='raise'))
        frame = raw.apply(_numeric)
        frame.index = index
        if not index.equals(dates):
            raise ValueError('Chickenpox timestamps differ from the original weekly grid.')
        provenance = 'original 522-week Hungary CSV'
    elif name == 'Influenza-US-2':
        required = {'REGION TYPE', 'REGION', 'YEAR', 'WEEK', 'ILITOTAL',
                    'NUM. OF PROVIDERS', 'TOTAL PATIENTS'}
        if not required.issubset(raw):
            raise ValueError('Missing CDC columns: ' + ', '.join(sorted(required - set(raw))))
        block = raw.loc[raw['REGION TYPE'].astype(str).str.strip() == country].copy()
        for key in ('YEAR', 'WEEK'):
            v = pd.to_numeric(block[key], errors='raise')
            if v.isna().any() or (v != np.floor(v)).any():
                raise ValueError('CDC YEAR and WEEK must be integers.')
            block[key] = v.astype(int)
        # CDC MMWR: Sunday-start week 1 contains January 4. Do not assign
        # artificial dates by row position: out-of-period exports must be dropped.
        block['date'] = [mmwr_start(y, w) for y, w in zip(block.YEAR, block.WEEK)]
        block = block.loc[block.date.between(dates[0], dates[-1])].copy()
        block['REGION'] = block['REGION'].astype(str).str.strip()
        columns = ['Region {}'.format(i) for i in range(1, 11)]
        if set(block.REGION) != set(columns) or not block.date.isin(dates).all():
            raise ValueError('Unexpected HHS geography or nonweekly dates.')
        if block.duplicated(['date', 'REGION']).any():
            raise ValueError('Duplicate HHS region/week reports.')
        # Benchmark incidence is a count, not the similarly sparse ILI percent.
        block['count'] = _numeric(block.ILITOTAL.replace({'X': np.nan, 'x': np.nan}))
        # In this CDC export, some historical weeks have zero providers, zero
        # patients and zero ILI visits in ALL ten HHS regions. This is a
        # surveillance outage, not ten concurrent genuine zero-incidence
        # observations. Keep the original raw target values and MASE scale,
        # but mark those rows as UNREPORTED so no training window can teach
        # the model to forecast artificial zero-incidence summers.
        providers = _numeric(block['NUM. OF PROVIDERS'])
        patients = _numeric(block['TOTAL PATIENTS'])
        if providers.isna().any() or patients.isna().any():
            raise ValueError('CDC reporting coverage fields must be numeric.')
        missing_report = (providers == 0) & (patients == 0) & (block['count'] == 0)
        if ((providers == 0) | (patients == 0)).ne(missing_report).any():
            raise ValueError('Ambiguous CDC zero-coverage report; check raw export.')
        block['reported'] = ~missing_report
        frame = block.pivot(index='date', columns='REGION', values='count').reindex(
            index=dates, columns=columns)
        report_mask = block.pivot(index='date', columns='REGION', values='reported').reindex(
            index=dates, columns=columns)
        coverage_matrix = block.pivot(
            index='date', columns='REGION', values='NUM. OF PROVIDERS').reindex(
                index=dates, columns=columns).to_numpy(dtype=np.float64)
        provenance = ('CDC HHS ILITOTAL original counts; weeks with zero providers '
                      'and zero patients are excluded from TRAINING windows only')
    else:
        required = {'adm_0_name', 'adm_1_name', 'adm_2_name', 'S_res', 'T_res',
                    'case_definition_standardised', 'calendar_start_date',
                    'calendar_end_date', 'dengue_total'}
        if not required.issubset(raw):
            raise ValueError('Missing original PAHO source columns: ' + ', '.join(sorted(required - set(raw))))
        for column in required - {'dengue_total'}:
            if raw[column].dtype == object:
                raw[column] = raw[column].map(lambda v: unicodedata.normalize('NFKC', v).strip()
                                              if isinstance(v, str) else v).replace('', np.nan)
        block = raw.loc[(raw.adm_0_name == country) & (raw.S_res == 'Admin2') &
                        (raw.T_res == 'Week') & (raw.case_definition_standardised == 'Total')].copy()
        block['date'] = pd.to_datetime(block.calendar_start_date, errors='raise')
        block['end'] = pd.to_datetime(block.calendar_end_date, errors='raise')
        if block[['date', 'end', 'adm_1_name', 'adm_2_name']].isna().any().any():
            raise ValueError('Missing municipality/province/week identity.')
        if not ((block.end - block.date).dt.days == 6).all():
            raise ValueError('Each PAHO weekly report must span exactly seven days.')
        block = block.loc[block.date.between(dates[0], dates[-1])].copy()
        if block.empty or not block.date.isin(dates).all():
            raise ValueError('No weekly reports or dates outside the Sunday grid.')
        if block.duplicated(['date', 'adm_1_name', 'adm_2_name']).any():
            raise ValueError('Duplicate municipality/week reports cannot be summed.')
        block['count'] = _numeric(block.dengue_total)
        grouped = block.groupby(['date', 'adm_1_name'])['count']
        # An explicitly missing municipal component makes its aggregate unknown.
        totals = grouped.sum(min_count=1).mask(grouped.count() != grouped.size())
        frame = totals.unstack('adm_1_name').sort_index(axis=1).reindex(dates)
        provenance = ('raw PAHO Admin2-to-province reconstruction; national/annual records excluded; '
                      'incomplete municipal reporting is not proven complete surveillance')
    if frame.shape != (length, channels) or frame.index.has_duplicates:
        raise ValueError('{} shape {} != ({}, {})'.format(name, frame.shape, length, channels))
    observed = frame.notna().to_numpy(dtype=bool)
    if name == 'Influenza-US-2':
        observed &= report_mask.fillna(False).to_numpy(dtype=bool)
    carried = frame.ffill()
    leading_gaps = int(carried.isna().to_numpy().sum())
    # LOCF never uses future observations. Leading gaps have no past report:
    # zero is an explicit reconstruction convention, not an observed zero count.
    values = carried.fillna(0.0).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Nonfinite or negative incidence after causal cleaning.')
    values.setflags(write=False)
    observed.setflags(write=False)
    if coverage_matrix is not None:
        if not np.isfinite(coverage_matrix).all() or (coverage_matrix < 0).any():
            raise ValueError('Invalid CDC reporting-provider counts.')
        coverage_matrix.setflags(write=False)
    return dates, tuple(map(str, frame.columns)), values, observed, leading_gaps, provenance, coverage_matrix


def read_four_source(name, root_path, data_path):
    if name not in FOUR_SPECS:
        raise ValueError('Unknown weekly epidemic dataset: ' + name)
    if data_path != FOUR_SPECS[name][0]:
        raise ValueError('Keep the original filename: ' + FOUR_SPECS[name][0])
    path = Path(root_path) / data_path
    if not path.is_file():
        raise FileNotFoundError('Place the extracted original CSV directly in dataset/: ' + str(path))
    stat = path.stat()
    return _read_cached(str(path.resolve()), name, stat.st_size, stat.st_mtime_ns)


class Dataset_EpiCastFixed(Dataset):
    """Sliding training samples, one fixed tail-H origin for each forecasting task."""
    def __init__(self, args, flag, refit=False):
        if flag not in ('train', 'val', 'test'):
            raise ValueError('Only train/val/test are supported by the fixed-origin adapter.')
        if args.features != 'M' or args.seq_len != 24 or args.label_len != 0:
            raise ValueError('Use features=M, seq_len=24 and label_len=0.')
        if args.pred_len not in (4, 8, 12) or args.cycle < 1:
            raise ValueError('Use H=4/8/12 and a positive cycle.')
        if args.data == 'Influenza-US-2' and args.target != 'ILITOTAL':
            raise ValueError('Influenza-US-2 predicts ILITOTAL counts.')
        if args.num_workers < 0 or args.batch_size < 1:
            raise ValueError('Invalid loader parameters.')
        self.name, self.flag = args.data, flag
        self.seq_len, self.pred_len, self.cycle = args.seq_len, args.pred_len, args.cycle
        self.chunk_len = args.pred_len
        (self.dates, self.columns, self.raw, self.observed,
         self.leading_gaps, self.provenance, self.coverage) = read_four_source(
             args.data, args.root_path, args.data_path)
        if len(self.columns) != args.enc_in:
            raise ValueError('enc_in differs from the number of source regions.')
        self.pretest_end = len(self.raw) - self.pred_len
        # Keep the entire held-out test tail inaccessible to model selection.
        # The short archive keeps two H=8 blocks for selection, leaving more
        # fitting history; H=4/12 retain 24 weeks (multiple H=12 backtests).
        validation_length = ({4: 24, 8: 16, 12: 24}[self.pred_len]
                             if self.name == 'Dengue-Argentina' else
                             max(12, self.cycle//2) if self.name == 'Dengue-Colombia' and self.pred_len == 4 else
                             max(12, 2*self.pred_len) if self.name == 'Influenza-US-2' else
                             max(12, int(np.ceil(0.10 * self.pretest_end))))
        self.validation_scheme = 'chronological-history-only'
        self.validation_length = validation_length
        self.selection_end = self.pretest_end - validation_length
        self.refit = bool(refit or flag == 'test')
        if flag == 'val' and self.refit:
            raise ValueError('Validation cannot use refitted statistics.')
        self.train_end = self.pretest_end if self.refit else self.selection_end
        if self.selection_end < self.seq_len + self.pred_len:
            raise ValueError('Not enough history for a complete training and validation chunk.')
        past = self.raw[:self.train_end]
        self.metric_mean = past.mean(axis=0)
        raw_std = past.std(axis=0)
        self.metric_std = np.where(raw_std > 1e-6, raw_std, 1.)
        self.train_scale = np.abs(np.diff(past, axis=0)).mean(axis=0)
        if not np.all(self.train_scale > 0):
            raise ValueError('MASE is undefined for a constant training channel.')
        if flag == 'train':
            self.origins = np.arange(self.seq_len, self.train_end - self.pred_len + 1,
                                     dtype=np.int64)
            if self.name == 'Influenza-US-2':
                # Sliding starts every week. Only windows containing a fully
                # reported 24-week input AND H-week target are trainable.
                # A masked week cannot silently become a zero-case label.
                all_reports = self.observed.all(axis=1)
                bad = np.concatenate(([0], np.cumsum(~all_reports)))
                starts = self.origins - self.seq_len
                ends = self.origins + self.pred_len
                self.origins = self.origins[bad[ends] == bad[starts]]
        elif flag == 'test':
            # Exactly one origin at T-H.  Test labels cannot enter any window.
            self.origins = np.array([self.pretest_end], dtype=np.int64)
        else:
            # Internal historical backtests may roll; final TEST NEVER rolls.
            self.origins = np.arange(self.selection_end,
                                     self.pretest_end - self.pred_len + 1,
                                     dtype=np.int64)
        if not len(self.origins):
            raise ValueError('The selected segment contains no complete forecast chunk.')
        if self.name == 'Influenza-US-2' and flag in ('val', 'test'):
            if not self.observed[np.concatenate([np.arange(o, o+self.pred_len) for o in self.origins])].all():
                raise ValueError('Held-out CDC targets contain unreported weeks.')
        self.marks = np.stack((self.dates.month, self.dates.dayofyear), axis=-1).astype(np.float32)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        end = origin + self.chunk_len
        sample = (self.raw[origin-self.seq_len:origin].astype(np.float32),
                  self.raw[origin:end].astype(np.float32),
                  self.marks[origin-self.seq_len:origin], self.marks[origin:end],
                  torch.tensor(origin % self.cycle, dtype=torch.long))
        if self.name == 'Influenza-US-2' and self.pred_len == 4:
            providers = self.coverage[origin-self.seq_len:origin].astype(np.float32)
            if not (providers > 0).all():
                raise ValueError('CDC H=4 exposure input has an unreported week.')
            return sample + (providers,)
        return sample


def fixed_data_provider(args, flag, refit=False):
    dataset = Dataset_EpiCastFixed(args, flag, refit)
    loader = DataLoader(dataset, batch_size=args.batch_size if flag != 'test' else 1, shuffle=flag == 'train',
                        drop_last=False, num_workers=args.num_workers)
    print('{} | {} | shape={} | windows={} | training_prefix={} | '
          'fixed_test_origin={} ({}) | training_output_H={} | leading_gap_initializations={} | {}'.format(
              args.data, flag + ('/refit' if dataset.refit else ''), dataset.raw.shape,
              len(dataset), dataset.train_end, dataset.pretest_end,
              dataset.dates[dataset.pretest_end].date(), args.pred_len,
              dataset.leading_gaps, dataset.provenance), flush=True)
    return dataset, loader


FILENAME = 'Influenza US.csv'
# 50 states, DC, Puerto Rico, Virgin Islands. NYC is a separate surveillance
# jurisdiction within New York; CNMI has no usable ILITOTAL in this export.
EXCLUDED = {'New York City', 'Commonwealth of the Northern Mariana Islands'}


def parse_influenza_us(stream):
    """Reconstruct the 521 x 53 grid; retain genuine zero case counts."""
    raw = pd.read_csv(stream, skiprows=1, encoding='utf-8-sig',
                      na_values=['X', 'x'], low_memory=False)
    raw.columns = raw.columns.str.strip()
    if raw.columns.duplicated().any():
        raise ValueError('Duplicate CDC headers.')
    required = {'REGION TYPE', 'REGION', 'YEAR', 'WEEK', 'ILITOTAL',
                'NUM. OF PROVIDERS', 'TOTAL PATIENTS'}
    if not required.issubset(raw):
        raise ValueError('Missing CDC fields: ' + ', '.join(sorted(required-set(raw))))
    raw['REGION'] = raw['REGION'].astype(str).str.strip()
    block = raw.loc[(raw['REGION TYPE'].astype(str).str.strip() == 'States') &
                    ~raw.REGION.isin(EXCLUDED)].copy()
    for field in ('YEAR', 'WEEK'):
        v = pd.to_numeric(block[field], errors='raise')
        if v.isna().any() or (v != np.floor(v)).any():
            raise ValueError('CDC year/week must be integers.')
        block[field] = v.astype(int)
    block['date'] = [mmwr_start(y, w) for y, w in zip(block.YEAR, block.WEEK)]
    dates = pd.date_range('2010-10-03', '2020-09-20', freq='7D')
    block = block.loc[block.date.between(dates[0], dates[-1])].copy()
    if block.duplicated(['date', 'REGION']).any() or not block.date.isin(dates).all():
        raise ValueError('Duplicate or misaligned CDC week/region.')
    for field in ('ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS'):
        v = pd.to_numeric(block[field], errors='raise')
        if np.isinf(v).any() or (v.dropna() < 0).any():
            raise ValueError('CDC values must be nonnegative and finite when observed.')
        block[field] = v
    columns = sorted(block.REGION.unique())
    if len(columns) != 53:
        raise ValueError('Expected 53 CDC jurisdictions, received {}'.format(len(columns)))
    def pivot(field):
        return block.pivot(index='date', columns='REGION', values=field).reindex(
            index=dates, columns=columns)
    cases, providers, patients = map(pivot, ('ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS'))
    outage = cases.eq(0) & providers.eq(0) & patients.eq(0)
    observed = cases.notna() & ~outage
    # These reconstructed RAW counts alone determine labels and MASE scale.
    # Leading absent records cannot be LOCF-filled; explicitly initialize zero.
    leading = int(cases.ffill().isna().to_numpy().sum())
    values = cases.ffill().fillna(0.).to_numpy(dtype=np.float64)
    # Missing surveillance is filled ONLY in model inputs, never in raw targets.
    inputs = cases.where(observed).ffill().fillna(0.).to_numpy(dtype=np.float64)
    valid = observed.to_numpy(dtype=bool)
    if values.shape != (521, 53) or not np.isfinite(values).all():
        raise ValueError('Invalid reconstructed Influenza US grid.')
    for array in (values, inputs, valid):
        array.setflags(write=False)
    return dates, tuple(columns), values, inputs, valid, leading


@lru_cache(maxsize=1)
def _read_influenza_us_cached(path, size, mtime_ns):
    return parse_influenza_us(path)


def read_influenza_us(root_path, data_path):
    if data_path != FILENAME:
        raise ValueError('Keep the original filename: ' + FILENAME)
    path = Path(root_path) / data_path
    if not path.is_file():
        raise FileNotFoundError('Place the original CSV directly in dataset/: ' + str(path))
    stat = path.stat()
    return _read_influenza_us_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


class Dataset_InfluenzaUS(Dataset):
    """Weekly sliding training; exactly one final origin T-H."""
    def __init__(self, args, flag, refit=False):
        if flag not in ('train', 'val', 'test') or args.data != 'Influenza-US':
            raise ValueError('Invalid Influenza US task/split.')
        if (args.features != 'M' or args.seq_len != 24 or args.label_len != 0 or
                args.target != 'ILITOTAL' or args.pred_len not in (4, 8, 12) or
                args.enc_in != 53 or args.cycle < 1):
            raise ValueError('Use M, 24 inputs, label_len=0, ILITOTAL, H=4/8/12 and enc_in=53.')
        if args.batch_size < 1 or args.num_workers < 0:
            raise ValueError('Invalid loader parameters.')
        self.name, self.flag = args.data, flag
        self.seq_len, self.pred_len, self.cycle = args.seq_len, args.pred_len, args.cycle
        self.chunk_len = args.pred_len
        (self.dates, self.columns, self.raw, self.inputs, self.observed,
         self.leading_gaps) = read_influenza_us(args.root_path, args.data_path)
        self.coverage = None
        self.pretest_end = len(self.raw) - self.pred_len
        # A full 12-week historical forecast block per H. Extending this
        # to 2H would put H=12 selection training entirely before the 2020
        # reporting shock, although that shock is known at the final origin.
        self.validation_length = 12
        self.selection_end = self.pretest_end - self.validation_length
        self.validation_scheme = 'chronological-history-only'
        self.refit = bool(refit or flag == 'test')
        if flag == 'val' and self.refit:
            raise ValueError('Validation cannot use refitted statistics.')
        self.train_end = self.pretest_end if self.refit else self.selection_end
        past = self.raw[:self.train_end]
        self.metric_mean = past.mean(axis=0)
        std = past.std(axis=0)
        self.metric_std = np.where(std > 1.e-6, std, 1.)
        self.train_scale = np.abs(np.diff(past, axis=0)).mean(axis=0)
        if not (self.train_scale > 0).all():
            raise ValueError('MASE undefined for a constant training series.')
        if flag == 'train':
            self.origins = np.arange(24, self.train_end-self.pred_len+1, dtype=np.int64)
        elif flag == 'val':
            self.origins = np.arange(self.selection_end, self.pretest_end-self.pred_len+1,
                                     dtype=np.int64)
        else:
            self.origins = np.array([self.pretest_end], dtype=np.int64)
        if flag != 'train' and not all(self.observed[o:o+self.pred_len].all() for o in self.origins):
            raise ValueError('Held-out CDC targets contain unreported values.')
        self.marks = np.stack((self.dates.month, self.dates.dayofyear), axis=-1).astype(np.float32)
        self.provenance = ('CDC state ILITOTAL; inferred 53-jurisdiction reconstruction; '
                           'causal input-only LOCF; unreported training targets masked')

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        o = int(self.origins[index])
        h = self.pred_len
        # The sixth tensor is a target mask. Only training loss consumes it;
        # inference forwards receive history and calendar phase alone.
        return (self.inputs[o-24:o].astype(np.float32),
                self.raw[o:o+h].astype(np.float32),
                self.marks[o-24:o], self.marks[o:o+h],
                torch.tensor(o % self.cycle, dtype=torch.long),
                self.observed[o:o+h].astype(np.float32))


def influenza_data_provider(args, flag, refit=False):
    dataset = Dataset_InfluenzaUS(args, flag, refit)
    loader = DataLoader(dataset, batch_size=1 if flag == 'test' else args.batch_size,
                        shuffle=flag == 'train', drop_last=False, num_workers=args.num_workers)
    print('{} | {} | shape={} | windows={} | training_prefix={} | fixed_test_origin={} ({}) | {}'.format(
        args.data, flag + ('/refit' if dataset.refit else ''), dataset.raw.shape,
        len(dataset), dataset.train_end, dataset.pretest_end,
        dataset.dates[dataset.pretest_end].date(), dataset.provenance), flush=True)
    return dataset, loader

EPICAST_DATASETS = tuple(FOUR_SPECS) + ('Influenza-US',)


def epicast_data_provider(args, flag, refit=False):
    """Route all five datasets to their unchanged sampling implementations."""
    if args.data == 'Influenza-US':
        return influenza_data_provider(args, flag, refit)
    if args.data in FOUR_SPECS:
        return fixed_data_provider(args, flag, refit)
    raise ValueError('Unknown EpiCast dataset: ' + args.data)
