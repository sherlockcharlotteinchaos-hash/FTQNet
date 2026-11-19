"""Experiments for all five weekly EpiCastBench datasets.

Shared fitting, refitting, checkpoint and reporting logic lives in this module.
Dataset-specific networks and losses retain the uploaded implementation.
Every final test contains exactly one origin T-H.
"""
import hashlib
import math
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_provider.data_epicast import fixed_data_provider, influenza_data_provider
from exp.exp_basic import Exp_Basic
from models.FTQNet import Model
from utils.tools import adjust_learning_rate, visual


PROTOCOL = 'epicast-single-tail-origin-v7-regional'
MODE = {
    'Chickenpox-Hungary': {4: 'global_trend', 8: 'global_last', 12: 'global_last'},
    'Dengue-Colombia': {4: 'global_last', 8: 'global_last', 12: 'global_last'},
    'Dengue-Argentina': {4: 'global_trend', 8: 'log_relative', 12: 'global_last'},
    'Influenza-US-2': {4: 'log_relative', 8: 'global_trend', 12: 'log_relative'},
}
# Allow an epidemic to restart after a constant window only for Argentina H4.
KEEP_CONSTANT = {'Dengue-Argentina': {4: False, 8: True, 12: True}}
RECENT_WEEKS = {'Chickenpox-Hungary': 156, 'Dengue-Colombia': 156,
                'Dengue-Argentina': 52, 'Influenza-US-2': 156}
INFERENCE_KEYS = ('data', 'data_path', 'features', 'target', 'seq_len',
                  'label_len', 'pred_len', 'enc_in', 'cycle', 'd_model',
                  'dropout', 'use_revin', 'model_type', 'flinear_alpha',
                  'flinear_c', 'flinear_groups')


def raw_metrics(prediction, truth, dataset):
    """MASE, SMAPE and RMSE: per-series first, equal average over series."""
    error = prediction - truth
    absolute = np.abs(error)
    denom = np.abs(prediction) + np.abs(truth)
    smape = np.divide(200. * absolute, denom, out=np.zeros_like(absolute),
                      where=denom > 0)
    return {'mase': float((absolute.mean(axis=(0, 1)) / dataset.train_scale).mean()),
            'raw_mae': float(absolute.mean()),
            'raw_rmse': float(np.sqrt(np.square(error).mean(axis=(0, 1))).mean()),
            'smape': float(smape.mean()),
            'mse': float(np.square(error / dataset.metric_std).mean()),
            'mae': float((absolute / dataset.metric_std).mean())}


def flu_constraints(prediction, history, horizon):
    """One causal inference map for historical validation and final prediction.

    All thresholds depend on the input window, never on future reports.
    Training uses raw-scale regression; learning through these hard limits
    was not robust across normal seasons and the 2020 historical shock.
    """
    recent = history[:, -4:, :]
    latest = recent[:, -1:, :]
    steps = torch.arange(1, horizon + 1, device=history.device,
                         dtype=history.dtype)[None, :, None]
    if horizon in (4, 8):
        delta = recent.diff(dim=1).abs().mean(dim=1, keepdim=True)
        band = steps * delta
        prediction = latest + torch.maximum(
            torch.minimum(prediction - latest, band), -band)
    prediction = prediction.clamp_min(0.)
    slope = (latest - recent[:, :1, :]) / 3.
    taper = (1. - .80 ** steps) / .20
    ceiling = (latest + .15 * slope.clamp_max(0.) * taper).clamp_min(0.)
    prediction = torch.where(slope < 0., torch.minimum(prediction, ceiling), prediction)
    if horizon == 4:
        return .50 * prediction + .50 * latest
    peak = history.amax(dim=1)
    level = recent.mean(dim=1)
    # With ten channels, median is the average of the middle two values.
    # torch.median would select only the lower middle value.
    shock = (torch.quantile(peak / level.clamp_min(1.), .5, dim=1) >= 2.5)[:, None, None]
    cap = torch.maximum(history[:, -8:, :].amax(dim=1, keepdim=True)
                        * (1.35 if horizon == 12 else 1.50), latest)
    prediction = torch.where(shock, torch.minimum(prediction, cap), prediction)
    excess = (prediction - recent.mean(dim=1, keepdim=True)).clamp_min(0.)
    decay = (.80 if horizon == 12 else .88) ** (steps - 1.)
    prediction = torch.where(shock, prediction - excess * (1. - decay), prediction)
    if horizon == 12:
        rise = (recent[:, -2:, :].mean(dim=1, keepdim=True)
                - recent[:, :2, :].mean(dim=1, keepdim=True)) > .10 * level[:, None, :].clamp_min(1.)
        floor = (latest * (1. - .05 * steps)).clamp_min(0.)
        prediction = torch.where(shock & rise & (steps <= 2),
                                 torch.maximum(prediction, floor), prediction)
    return prediction


# Used only on Dengue Colombia H=8/12. Candidate/confirmation periods
# must be entirely inside the historical validation interval.
def shared_channel_anchors(dataset):
    """Deterministic flu-inspired, 24-week-causal levels for channel shrinkage.

    A long-run seasonal mean is fitted entirely before the historical validation
    segment and is re-fitted before the final held-out tail at inference time.
    It is a fixed prefix statistic, not access to unobserved future values.
    """
    origins = dataset.origins
    H, C = dataset.pred_len, len(dataset.columns)
    histories = np.stack([dataset.raw[o-24:o] for o in origins])
    recent = histories[:, -4:, :]
    last = recent[:, -1:, :]
    step = np.arange(1, H + 1, dtype=np.float64)[None, :, None]
    slope = (recent[:, -1:, :] - recent[:, :1, :]) / 3.
    trend = np.maximum(last + slope * (.8 * (1. - .8 ** step) / .2), 0.)
    choices = {'last': np.broadcast_to(last, (len(origins), H, C)),
               'mean4': np.broadcast_to(recent.mean(axis=1, keepdims=True),
                                        (len(origins), H, C)),
               'trend': trend,
               'ema': np.broadcast_to(
                   .5 * last + .3 * recent[:, -2:-1, :] + .2 * recent[:, -3:-2, :],
                   (len(origins), H, C))}
    if dataset.train_end >= 3 * dataset.cycle:
        # Historical prefix only: no validation or test data enter this profile.
        fit_history = dataset.raw[:dataset.train_end]
        ts = np.arange(dataset.train_end)
        profile = np.empty((dataset.cycle, C), dtype=np.float64)
        for phase in range(dataset.cycle):
            offset = (ts - phase) % dataset.cycle
            near = np.minimum(offset, dataset.cycle - offset) <= 2
            profile[phase] = np.median(fit_history[near], axis=0)
        q_future = (origins[:, None] + np.arange(H)[None, :]) % dataset.cycle
        q_recent = (origins[:, None] - 4 + np.arange(4)[None, :]) % dataset.cycle
        recent_template = profile[q_recent].mean(axis=1, keepdims=True)
        seasonal = np.maximum(profile[q_future] *
                              (recent.mean(axis=1, keepdims=True) /
                               np.maximum(recent_template, 1.)), 0.)
        choices['seasonal'] = seasonal
    return choices


def apply_channel_calibration(prediction, dataset, entries):
    """Apply frozen per-channel shrinkage to a past-data anchor."""
    if len(entries) != len(dataset.columns):
        raise ValueError('Frozen channel-calibration vector differs from source regions.')
    if not any(entry is not None for entry in entries):
        return prediction
    anchors = shared_channel_anchors(dataset)
    out = np.asarray(prediction, dtype=np.float64).copy()
    for ch, entry in enumerate(entries):
        if entry is None:
            continue
        name, weight = entry
        if name not in anchors or weight not in (.25, .50, .75):
            raise ValueError('Invalid historical channel-calibration value.')
        out[:, :, ch] = (weight * out[:, :, ch] +
                         (1. - weight) * anchors[name][:, :, ch])
    return np.maximum(out, 0.)


def fit_channel_calibration(dataset, prediction, truth):
    """Cross-check each channel on *later* historical validation weeks.

    Candidate/weight discovery uses only the first half of validation origins.
    The second half must independently support a channel intervention.  Short
    dengue Argentina histories are deliberately left uncalibrated.
    """
    n, H, C = prediction.shape
    baseline = float((np.abs(prediction - truth) /
                      dataset.train_scale[None, None, :]).mean())
    if n < 32:
        return [None] * C, baseline, baseline
    anchors = shared_channel_anchors(dataset)
    cut = n // 2
    selected = [None] * C
    for ch in range(C):
        y = truth[:, :, ch]
        p = prediction[:, :, ch]
        early_base = np.abs(p[:cut] - y[:cut]).mean()
        late_base = np.abs(p[cut:] - y[cut:]).mean()
        if early_base <= 1.e-9:
            continue
        best_candidate, best_error = None, early_base
        for name, anchor in anchors.items():
            x = anchor[:, :, ch]
            for weight in (.25, .50, .75):
                candidate = weight * p + (1. - weight) * x
                err = np.abs(candidate[:cut] - y[:cut]).mean()
                if err < best_error:
                    best_error, best_candidate = err, (name, weight)
        if best_candidate is None or best_error >= .96 * early_base:
            continue
        name, weight = best_candidate
        late = np.abs((weight * p[cut:] +
                       (1. - weight) * anchors[name][cut:, :, ch]) - y[cut:]).mean()
        # Require independently validated late-fold gains; don't sacrifice
        # the last validation block to a seasonal rule that fits only the past.
        if late <= .98 * late_base:
            selected[ch] = [name, weight]
    adjusted = apply_channel_calibration(prediction, dataset, selected)
    score = float((np.abs(adjusted - truth) /
                   dataset.train_scale[None, None, :]).mean())
    # Rolling validation windows overlap heavily. An improvement <3% may
    # reflect a few correlated weeks rather than durable new information.
    if score >= .97 * baseline:
        selected = [None] * C
        score = baseline
    print('Historical-only channel calibration | dataset={} H={} validated_channels={}/{} '
          'raw_MASE={:.6f} calibrated_MASE={:.6f}'.format(
              dataset.name, H, sum(x is not None for x in selected), C,
              baseline, score), flush=True)
    return selected, baseline, score




class ForecastNetwork(torch.nn.Module):
    """Dataset-specific CAUSAL coordinate transforms around unchanged FTQNet."""

    def __init__(self, args, fit_history, fit_observed=None, fit_exposure=None):
        super().__init__()
        self.horizon = args.pred_len
        self.is_flu = args.data == 'Influenza-US-2'
        self.flu_rate_mode = args.data == 'Influenza-US-2' and args.pred_len == 4
        self.mode = MODE[args.data][args.pred_len]
        self.keep_constant = KEEP_CONSTANT.get(args.data, {}).get(args.pred_len, True)
        self.net = Model(args).float()
        if args.data == 'Influenza-US-2' and args.pred_len == 12:
            # For this sparse regime-shift target, 0.5 attention dropout was
            # found unstable on historical held-out backtests. Override only
            # the private model instance for this task: never the shared Model.
            self.net.channelAggregator.dropout = 0.0
        valid_rows = (np.asarray(fit_observed, dtype=bool).all(axis=1)
                      if args.data == 'Influenza-US-2' and fit_observed is not None
                      else np.ones(len(fit_history), dtype=bool))
        if not valid_rows.any():
            raise ValueError('No fully reported CDC weeks in the model prefix.')
        if self.flu_rate_mode:
            if fit_exposure is None or fit_exposure.shape != fit_history.shape:
                raise ValueError('CDC H=4 requires aligned historical provider counts.')
            fit_history = fit_history * 10. / np.maximum(fit_exposure, 1.)
        if self.is_flu and self.horizon == 8:
            # Input-activity balancing is fitted on reported HISTORY only.
            # H=8 is in raw-count coordinates, so this is the raw case level.
            level = fit_history[valid_rows].mean(axis=0)
            self.register_buffer('loss_level', torch.tensor(level, dtype=torch.float32).clamp_min(1.))
        valid_history = fit_history[valid_rows]
        mean = valid_history.mean(axis=0)
        std = valid_history.std(axis=0)
        std = np.where(std > 1.e-6, std, 1.)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        with torch.no_grad():
            # The residual head starts near a strong simple forecasting anchor.
            self.net.output_proj[-1].weight.zero_()
            self.net.output_proj[-1].bias.zero_()
            if args.data in ('Chickenpox-Hungary', 'Influenza-US-2'):
                # Prefix-only seasonal profile initializes an EXISTING FTQNet
                # parameter; no external pretraining and no extra parameters.
                z = (fit_history - mean) / std
                phase = np.arange(len(z)) % args.cycle
                profile = np.stack([
                    z[(phase == j) & valid_rows].mean(axis=0) if np.any((phase == j) & valid_rows)
                    else np.zeros(z.shape[1], dtype=np.float64)
                    for j in range(args.cycle)])
                profile = sum(np.roll(profile, d, axis=0)
                              for d in (-2, -1, 0, 1, 2)) / 5.
                self.net.temporalQuery.copy_(torch.tensor(profile, dtype=torch.float32))

    def forward(self, history, phase, exposure=None):
        raw_history = history
        if self.flu_rate_mode:
            if exposure is None or exposure.shape != history.shape:
                raise ValueError('CDC H=4 requires historical provider counts.')
            last_exposure = exposure[:, -1:, :]
            history = history * 10. / exposure.clamp_min(1.)
        if self.mode.startswith('global'):
            z = (history - self.mean) / self.std
            z = z - z.mean(dim=1, keepdim=True)
            anchor = history[:, -1:, :]
            if self.mode == 'global_trend':
                # A small, decaying local-trend prior in the raw scale.
                t = torch.arange(6, device=history.device, dtype=history.dtype) - 2.5
                slope = (history[:, -6:, :] * t[None, :, None]).sum(
                    dim=1, keepdim=True) / t.square().sum()
                lead = torch.arange(1, self.horizon + 1,
                                    device=history.device, dtype=history.dtype)[None, :, None]
                anchor = (anchor + slope * (.8 * (1. - .8 ** lead) / .2)).clamp_min(0.)
            forecast = anchor + self.net(z, phase) * self.std
        elif self.mode == 'log_relative':
            z = torch.log1p(history.clamp_min(0.))
            scale = z.std(dim=1, keepdim=True, unbiased=False).clamp_min(.25)
            centered = (z - z.mean(dim=1, keepdim=True)) / scale
            baseline = z[:, -4:, :].mean(dim=1, keepdim=True)
            # expm1 is differentiable; raw target/loss is never log-transformed.
            forecast = torch.expm1((baseline + self.net(centered, phase) * scale).clamp(max=20.))
        elif self.mode == 'relative':
            scale = history.abs().mean(dim=1, keepdim=True).clamp_min(1.)
            z = (history - history.mean(dim=1, keepdim=True)) / scale
            forecast = history[:, -4:, :].mean(dim=1, keepdim=True) + self.net(z, phase) * scale
        else:
            raise ValueError('Unsupported epidemic coordinate mode: ' + self.mode)
        if self.keep_constant:
            constant = history.amax(dim=1, keepdim=True) == history.amin(dim=1, keepdim=True)
            forecast = torch.where(constant, history[:, -1:, :], forecast)
        if self.flu_rate_mode:
            forecast = forecast * (last_exposure / 10.)
        if self.is_flu and not self.training:
            forecast = flu_constraints(forecast, raw_history, self.horizon)
        # Keep regression gradients before hard inference limits in training.
        return forecast


class Exp_EpiCastFixed(Exp_Basic):
    # Default factories retain the original four-task numerical path.
    data_factory = staticmethod(fixed_data_provider)
    network_factory = staticmethod(ForecastNetwork)
    coordinate_modes = MODE
    recent_weeks = RECENT_WEEKS
    protocol = PROTOCOL

    def __init__(self, args):
        self._splits = {}
        self.selection = None
        self.source_sha256 = None
        self._refitting = False
        super().__init__(args)

    def _seed(self):
        random.seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.random_seed)

    def _get_data(self, flag, refit=False):
        key = (flag, bool(refit or flag == 'test'))
        if key not in self._splits:
            self._splits[key] = self.data_factory(self.args, flag, refit)
        return self._splits[key]

    def _build_model(self):
        a = self.args
        if a.use_multi_gpu or a.use_amp or a.test_flop or a.itr != 1:
            raise ValueError('These epidemic tasks require float32, one device and itr=1.')
        if a.loss != 'mae' or a.train_epochs < 1 or a.patience < 1 or a.learning_rate <= 0.:
            raise ValueError('Use --loss mae and positive training budgets/learning rate.')
        if a.lradj not in ('constant', 'type1', 'type2', 'type3', 'cosine',
                           '3', '4', '5', '6', 'TST'):
            raise ValueError('Unsupported learning rate schedule.')
        train, _ = self._get_data('train')
        return self.network_factory(a, train.raw[:train.train_end],
                               train.observed[:train.train_end],
                               None if train.coverage is None else train.coverage[:train.train_end]).float()

    def _fingerprint(self):
        digest = hashlib.sha256()
        with (Path(self.args.root_path) / self.args.data_path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _collect(self, flag):
        if flag not in ('val', 'test'):
            raise ValueError('Only historical validation or fixed-origin test is permitted.')
        dataset, loader = self._get_data(flag)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in loader:
                history, _, _, _, phase, *extra = batch
                exposure = (extra[0].to(self.device, dtype=torch.float32)
                            if extra else None)
                pred = self.model(history.to(self.device, dtype=torch.float32),
                                  phase.to(self.device, dtype=torch.long), exposure)
                predictions.append(pred.cpu().numpy().astype(np.float64))
        pred = np.maximum(np.concatenate(predictions, axis=0), 0.)
        if (flag == 'test' and self.args.data == 'Dengue-Colombia'
                and self.args.pred_len in (8, 12) and self.selection is not None
                and 'channel_calibration' in self.selection):
            pred = apply_channel_calibration(
                pred, dataset, self.selection['channel_calibration'])
        origins = dataset.origins
        true = np.stack([dataset.raw[o:o+dataset.pred_len] for o in origins])
        if pred.shape != true.shape or not np.isfinite(pred).all():
            raise FloatingPointError('Nonfinite or mismatched prediction shape.')
        if flag == 'test' and (len(origins) != 1 or origins[0] != len(dataset.raw)-dataset.pred_len):
            raise RuntimeError('Fixed-origin test must consist of exactly T-H.')
        return dataset, pred, true

    def _fit_stage(self, dataset, loader, epochs, rate, stage, validate):
        if epochs < 1:
            raise ValueError('Each fitting stage must contain a positive number of epochs.')
        optim = torch.optim.Adam(self.model.parameters(), lr=rate)
        config = SimpleNamespace(**vars(self.args))
        config.learning_rate = rate
        scheduler = None
        if config.lradj == 'TST':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim, max_lr=rate, epochs=epochs, steps_per_epoch=len(loader),
                pct_start=config.pct_start)
        denom = torch.as_tensor(dataset.train_scale, device=self.device, dtype=torch.float32)
        best, best_epoch, best_state, stale = float('inf'), 0, None, 0
        for epoch in range(1, epochs+1):
            self.model.train()
            total, samples = 0., 0
            for batch in loader:
                x, y, _, _, phase, *extra = batch
                exposure = (extra[0].to(self.device, dtype=torch.float32)
                            if extra else None)
                optim.zero_grad(set_to_none=True)
                forecast = self.model(x.to(self.device, dtype=torch.float32),
                                      phase.to(self.device, dtype=torch.long), exposure)
                errors = (forecast - y.to(self.device, dtype=torch.float32)).abs() / denom
                if self.args.data == 'Influenza-US-2' and self.args.pred_len == 8:
                    level = self.model.loss_level[None, None, :]
                    activity = x.to(self.device, dtype=torch.float32).mean(dim=1, keepdim=True)
                    weight = (level / (activity + .10 * level).clamp_min(1.)).clamp(.25, 4.)
                    loss = (errors * weight).mean()
                else:
                    loss = errors.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite MASE-weighted training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optim.step()
                total += float(loss.item()) * len(x)
                samples += len(x)
                if scheduler is not None:
                    scheduler.step()
            if validate:
                if self._refitting:
                    raise RuntimeError('Validation labels are forbidden during full-prefix refit.')
                val, p, y = self._collect('val')
                score = raw_metrics(p, y, val)['mase']
                if not math.isfinite(score):
                    raise FloatingPointError('Nonfinite validation MASE.')
                if score < best - 1.e-8:
                    best, best_epoch, stale = score, epoch, 0
                    best_state = {key: val.detach().cpu().clone()
                                  for key, val in self.model.state_dict().items()}
                else:
                    stale += 1
                print('{} epoch={}/{} train_MASE={:.6f} validation_MASE={:.6f} best={:.6f}'.format(
                    stage, epoch, epochs, total/samples, score, best), flush=True)
                if stale >= self.args.patience:
                    break
            else:
                print('{} epoch={}/{} train_MASE={:.6f}'.format(
                    stage, epoch, epochs, total/samples), flush=True)
            if scheduler is None:
                adjust_learning_rate(optim, None, epoch, config, printout=False)
        return best, best_epoch, best_state

    def _recent_loader(self, dataset):
        span = self.recent_weeks[self.args.data]
        cutoff = max(dataset.seq_len, dataset.train_end - span + dataset.seq_len)
        indices = np.flatnonzero(dataset.origins >= cutoff).tolist()
        if not indices:
            return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                              num_workers=self.args.num_workers, drop_last=False)
        return DataLoader(Subset(dataset, indices), batch_size=self.args.batch_size,
                          shuffle=True, num_workers=self.args.num_workers, drop_last=False)

    def train(self, setting):
        start = time.time()
        source_hash = self._fingerprint()
        train, loader = self._get_data('train')
        self._get_data('val')
        self._seed()
        self.model = self.network_factory(self.args, train.raw[:train.train_end],
                                     train.observed[:train.train_end],
                                     None if train.coverage is None else train.coverage[:train.train_end]).to(self.device)
        fine_budget = self.args.train_epochs // 5
        history_budget = self.args.train_epochs - fine_budget
        best, history_epochs, history_state = self._fit_stage(
            train, loader, history_budget, self.args.learning_rate,
            'history/selection', validate=True)
        if history_epochs < 1:
            raise RuntimeError('Historical selection did not finish an epoch.')
        selection = {'history_epochs': history_epochs, 'recent_epochs': 0,
                     'validation_mase': best}
        # The history checkpoint is kept only in memory. Fine-tuning is
        # selected solely on PAST validation weeks and is optional.
        self.model.load_state_dict(history_state, strict=True)
        if fine_budget:
            fine_score, fine_epoch, fine_state = self._fit_stage(
                train, self._recent_loader(train), fine_budget,
                self.args.learning_rate * .25, 'recent/selection', validate=True)
            if fine_epoch > 0 and fine_score < best - 1.e-8:
                selection.update(recent_epochs=fine_epoch, validation_mase=fine_score)
        # Reuse the historically supported regional correction from the
        # previous project, without changing the Improved model/training
        # choices or adding another CLI option. Fit only on pre-test weeks.
        if self.args.data == 'Dengue-Colombia' and self.args.pred_len in (8, 12):
            if selection['recent_epochs']:
                self.model.load_state_dict(fine_state, strict=True)
            else:
                self.model.load_state_dict(history_state, strict=True)
            val_data, val_prediction, val_truth = self._collect('val')
            entries, raw_score, calibration_score = fit_channel_calibration(
                val_data, val_prediction, val_truth)
            selection['channel_calibration'] = entries
            selection['channel_calibration_val'] = calibration_score
            selection['channel_uncalibrated_val'] = raw_score
        self.selection = selection
        self._refitting = True
        full, full_loader = self._get_data('train', refit=True)
        self._seed()
        self.model = self.network_factory(self.args, full.raw[:full.train_end],
                                     full.observed[:full.train_end],
                                     None if full.coverage is None else full.coverage[:full.train_end]).to(self.device)
        self._fit_stage(full, full_loader, history_epochs, self.args.learning_rate,
                        'history/refit', validate=False)
        if selection['recent_epochs']:
            self._fit_stage(full, self._recent_loader(full), selection['recent_epochs'],
                            self.args.learning_rate * .25, 'recent/refit', validate=False)
        if source_hash != self._fingerprint():
            raise ValueError('The raw source CSV changed during fitting.')
        path = Path(self.args.checkpoints) / setting
        path.mkdir(parents=True, exist_ok=True)
        torch.save({'protocol': self.protocol, 'forecast_horizon': self.args.pred_len,
                    'origin': int(full.pretest_end),
                    'model': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    'selection': selection, 'mode': self.coordinate_modes[self.args.data][self.args.pred_len],
                    'source_sha256': source_hash, 'train_end': full.train_end,
                    'selection_end': full.selection_end, 'validation_scheme': full.validation_scheme,
                    'columns': list(full.columns), 'fixed_origin': str(full.dates[full.pretest_end].date()),
                    'metric_mean': full.metric_mean.tolist(), 'metric_std': full.metric_std.tolist(),
                    'mase_scale': full.train_scale.tolist(),
                    'inference_config': {k: getattr(self.args, k) for k in INFERENCE_KEYS}},
                   path / 'checkpoint.pth')
        self.source_sha256 = source_hash
        print('Frozen selection={} | mode={} | train_end={} | seconds={:.1f}'.format(
            selection, self.coordinate_modes[self.args.data][self.args.pred_len], full.train_end, time.time()-start), flush=True)
        return self.model

    def _load_checkpoint(self, setting):
        # torch 2.4+ supports weights_only=True; checkpoint consists of safe
        # primitive types and tensors, not arbitrary pickle objects.
        ckpt = torch.load(Path(self.args.checkpoints) / setting / 'checkpoint.pth',
                          map_location=self.device, weights_only=True)
        full, _ = self._get_data('train', refit=True)
        if ckpt.get('protocol') != self.protocol or ckpt.get('forecast_horizon') != self.args.pred_len:
            raise ValueError('Checkpoint forecast protocol differs from this run.')
        if ckpt.get('source_sha256') != self._fingerprint():
            raise ValueError('Checkpoint source CSV changed.')
        if ckpt.get('mode') != self.coordinate_modes[self.args.data][self.args.pred_len]:
            raise ValueError('Checkpoint uses a different coordinate transform.')
        config = ckpt.get('inference_config', {})
        if set(config) != set(INFERENCE_KEYS) or any(
                config[k] != getattr(self.args, k) for k in INFERENCE_KEYS):
            raise ValueError('Inference parameter mismatch with saved checkpoint.')
        if (ckpt.get('origin') != full.pretest_end or ckpt.get('train_end') != full.train_end
                or ckpt.get('selection_end') != full.selection_end
                or ckpt.get('validation_scheme') != full.validation_scheme
                or ckpt.get('columns') != list(full.columns)
                or ckpt.get('fixed_origin') != str(full.dates[full.pretest_end].date())):
            raise ValueError('Checkpoint time split or regional series differ.')
        for key, expected in (('metric_mean', full.metric_mean),
                              ('metric_std', full.metric_std), ('mase_scale', full.train_scale)):
            if not np.array_equal(np.asarray(ckpt[key]), expected):
                raise ValueError('Checkpoint training-only statistic mismatch: ' + key)
        self.model = self.network_factory(self.args, full.raw[:full.train_end],
                                     full.observed[:full.train_end],
                                     None if full.coverage is None else full.coverage[:full.train_end]).to(self.device)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.selection, self.source_sha256, self._refitting = (
            ckpt['selection'], ckpt['source_sha256'], True)

    def test(self, setting, test=0):
        if test:
            self._load_checkpoint(setting)
        if self.selection is None or self.source_sha256 != self._fingerprint():
            raise ValueError('Training or loading the frozen checkpoint is required.')
        dataset, pred, true = self._collect('test')
        scores = raw_metrics(pred, true, dataset)
        if not all(math.isfinite(v) for v in scores.values()):
            raise FloatingPointError('Nonfinite final test metric.')
        # Exactly the same output names/types as the existing experiment code:
        # checkpoint.pth, result.txt, results/<dataset>.txt, test_results/<setting>/0.pdf.
        result_setting = re.sub(r'_fa[^_]+_fc[^_]+_fg[^_]+(?=_seed)', '', setting)
        folder = Path('test_results') / result_setting
        folder.mkdir(parents=True, exist_ok=True)
        origin = int(dataset.origins[0])
        history = ((dataset.raw[origin-dataset.seq_len:origin, -1]
                    - dataset.metric_mean[-1]) / dataset.metric_std[-1])
        vis_pred = (pred[0, :, -1]-dataset.metric_mean[-1])/dataset.metric_std[-1]
        vis_true = (true[0, :, -1]-dataset.metric_mean[-1])/dataset.metric_std[-1]
        visual(np.concatenate((history, vis_true)), np.concatenate((history, vis_pred)),
               str(folder / '0.pdf'))
        plt.close('all')
        obs = dataset.observed[origin:origin+dataset.pred_len]
        print('PROTOCOL={} | origin={} | windows=1 | H={} | reported_targets={}/{} | selection={}'.format(
            self.protocol, dataset.dates[origin].date(), dataset.pred_len,
            int(obs.sum()), obs.size, self.selection), flush=True)
        print('EpiCast raw metrics | MAE:{raw_mae:.10f}, RMSE:{raw_rmse:.10f}, '
              'SMAPE:{smape:.10f}, MASE:{mase:.10f}'.format(**scores), flush=True)
        print('mse:{}, mae:{}'.format(scores['mse'], scores['mae']), flush=True)
        with open('result.txt', 'a', encoding='utf-8') as handle:
            handle.write(result_setting + '  \n' +
                         'mse:{}, mae:{},mase:{},MAE:{}\n\n'.format(scores['mse'], scores['mae'],scores["mase"],scores["raw_mae"]))
        return scores


class StateForecastNetwork(torch.nn.Module):
    """Local raw/log coordinates and a causal residual forecasting head."""
    def __init__(self, args, fit_history, fit_observed=None, fit_exposure=None):
        super().__init__()
        self.horizon = args.pred_len
        self.net = Model(args).float()
        valid = np.ones_like(fit_history, dtype=bool) if fit_observed is None else fit_observed
        # Prefix-only statistics, per channel; never discard all 53 series
        # because one jurisdiction was absent in early surveillance years.
        if not valid.any(axis=0).all():
            raise ValueError('Every jurisdiction needs an observed training history.')
        level = (fit_history * valid).sum(axis=0) / valid.sum(axis=0)
        self.register_buffer('loss_level', torch.tensor(level, dtype=torch.float32).clamp_min(1.))
        z = np.log1p(fit_history)
        mean = (z*valid).sum(axis=0)/valid.sum(axis=0)
        var = (np.square(z-mean)*valid).sum(axis=0)/valid.sum(axis=0)
        std = np.maximum(np.sqrt(var), .25)
        with torch.no_grad():
            self.net.output_proj[-1].weight.zero_()
            self.net.output_proj[-1].bias.zero_()
            # Query indexing starts at the forecast origin in the shared model.
            phase = (np.arange(len(z)) + args.seq_len) % args.cycle
            normalized = (z-mean)/std
            profile = np.zeros((args.cycle, args.enc_in), dtype=np.float64)
            for j in range(args.cycle):
                mask = valid & (phase == j)[:, None]
                profile[j] = (normalized*mask).sum(axis=0)/np.maximum(mask.sum(axis=0), 1)
            profile = sum(np.roll(profile, j, axis=0) for j in (-2,-1,0,1,2))/5.
            self.net.temporalQuery.copy_(torch.tensor(profile, dtype=torch.float32))

    def forward(self, history, phase):
        if self.horizon == 4:
            # State-level short-range reports are noisier than HHS aggregates.
            # Regress additive residuals around a four-report arithmetic mean;
            # do not force the forecast back to a possibly isolated last spike.
            scale = history.mean(dim=1,keepdim=True).clamp_min(1.)
            centered = (history-history.mean(dim=1,keepdim=True))/scale
            recent_scale = history[:, -8:, :].mean(dim=1,keepdim=True).clamp_min(1.)
            anchor = history[:, -4:, :].mean(dim=1,keepdim=True)
            prediction = anchor+self.net(centered,phase)*recent_scale
            return prediction if self.training else prediction.clamp_min(0.)
        z = torch.log1p(history.clamp_min(0.))
        scale = z.std(dim=1, keepdim=True, unbiased=False).clamp_min(.25)
        centered = (z-z.mean(dim=1, keepdim=True))/scale
        # Keep the latest known reporting level. For H=12 use the existing
        # four-task damped trend, computed entirely within the input window.
        anchor = history[:, -1:, :]
        if self.horizon == 12:
            slope = (history[:, -1:, :]-history[:, -4:-3, :])/3.
            lead = torch.arange(1,13,device=history.device,dtype=history.dtype)[None,:,None]
            anchor = (anchor+slope*.8*(1.-.8**lead)/.2).clamp_min(0.)
        recent_scale = z[:, -8:, :].std(dim=1,keepdim=True,unbiased=False).clamp_min(.25)
        prediction = torch.expm1((torch.log1p(anchor)+self.net(centered, phase)*recent_scale).clamp(max=20.))
        return prediction if self.training else flu_constraints(prediction,history,self.horizon)


class Exp_InfluenzaUS(Exp_EpiCastFixed):
    data_factory = staticmethod(influenza_data_provider)
    network_factory = staticmethod(StateForecastNetwork)
    coordinate_modes = {'Influenza-US': {4: 'state_relative_mean', 8: 'state_log_last',
                                       12: 'state_log_trend'}}
    recent_weeks = {'Influenza-US': 156}
    protocol = 'epicast-state-single-tail-origin-v2'

    def _collect(self, flag):
        if flag not in ('val', 'test'):
            raise ValueError('Only historical validation or fixed-origin test is permitted.')
        dataset, loader = self._get_data(flag)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for history, _, _, _, phase, _ in loader:
                p = self.model(history.to(self.device,dtype=torch.float32),
                               phase.to(self.device,dtype=torch.long))
                predictions.append(p.cpu().numpy().astype(np.float64))
        prediction = np.maximum(np.concatenate(predictions,axis=0),0.)
        truth = np.stack([dataset.raw[o:o+dataset.pred_len] for o in dataset.origins])
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise FloatingPointError('Invalid state forecast.')
        if flag == 'test' and (len(dataset.origins) != 1 or dataset.origins[0] != len(dataset.raw)-dataset.pred_len):
            raise RuntimeError('Final test must have exactly one origin T-H.')
        return dataset,prediction,truth

    def _fit_stage(self, dataset, loader, epochs, rate, stage, validate):
        if epochs < 1:
            raise ValueError('A fitting stage must have positive epochs.')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=rate)
        config = SimpleNamespace(**vars(self.args))
        config.learning_rate = rate
        scheduler = None
        if config.lradj == 'TST':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=rate, epochs=epochs, steps_per_epoch=len(loader),
                pct_start=config.pct_start)
        denom = torch.tensor(dataset.train_scale, device=self.device, dtype=torch.float32)
        best, best_epoch, best_state, stale = float('inf'), 0, None, 0
        for epoch in range(1, epochs+1):
            self.model.train()
            total, samples = 0., 0
            for x, y, _, _, phase, mask in loader:
                x, y, mask = [v.to(self.device, dtype=torch.float32) for v in (x,y,mask)]
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(x, phase.to(self.device, dtype=torch.long))
                error = (prediction-y).abs()/denom
                # Preserve per-jurisdiction equality even when a territory has
                # fewer reports. Missing reports never become zero-case labels.
                counts = mask.sum(dim=(0,1))
                supported = counts > 0
                level = self.model.loss_level[None,None,:]
                activity = x.mean(dim=1,keepdim=True)
                weight = (level/(activity+.10*level).clamp_min(1.)).clamp(.25,4.)
                per_channel = (error*mask*weight).sum(dim=(0,1))/counts.clamp_min(1.)
                loss = per_channel[supported].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite state-level training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
                total += loss.item()*len(x)
                samples += len(x)
                if scheduler is not None:
                    scheduler.step()
            if validate:
                if self._refitting:
                    raise RuntimeError('Validation forbidden during full-prefix refit.')
                val,p,y = self._collect('val')
                score = raw_metrics(p,y,val)['mase']
                if not math.isfinite(score):
                    raise FloatingPointError('Nonfinite validation MASE.')
                if score < best-1.e-8:
                    best,best_epoch,stale = score,epoch,0
                    best_state = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                else:
                    stale += 1
                print('{} epoch={}/{} train_MASE={:.6f} validation_MASE={:.6f} best={:.6f}'.format(
                    stage,epoch,epochs,total/samples,score,best),flush=True)
                if stale >= self.args.patience:
                    break
            else:
                print('{} epoch={}/{} train_MASE={:.6f}'.format(
                    stage,epoch,epochs,total/samples),flush=True)
            if scheduler is None:
                adjust_learning_rate(optimizer,None,epoch,config,printout=False)
        return best,best_epoch,best_state

_EPICAST_EXPERIMENTS = {
    'Chickenpox-Hungary': Exp_EpiCastFixed,
    'Dengue-Colombia': Exp_EpiCastFixed,
    'Dengue-Argentina': Exp_EpiCastFixed,
    'Influenza-US-2': Exp_EpiCastFixed,
    'Influenza-US': Exp_InfluenzaUS,
}


def get_epicast_experiment(name):
    """Select the dataset-specific strategy within the unified experiment module."""
    if name not in _EPICAST_EXPERIMENTS:
        raise ValueError('Unknown EpiCast dataset: ' + name)
    return _EPICAST_EXPERIMENTS[name]
