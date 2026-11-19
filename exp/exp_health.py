"""Training and evaluation for COVID-19, ILI and Wearable.

The original data loaders, FTQNet model, loss, optimizer and learning-rate
schedule are preserved. Best model state and training history are kept in RAM;
Health and Wearable use the master project checkpoint and result format.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from utils.tools import visual
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.data_health import (
    health_data_provider, COVID_GROUPS as GROUPS,
    covid_group_indices as group_indices, covid_training_zero_floor,
)
from exp.exp_basic import Exp_Basic
from models.FTQNet import Model


class Exp_Health(Exp_Basic):
    def _acquire_device(self):
        device = torch.device('cuda:{}'.format(self.args.gpu) if self.args.use_gpu else 'cpu')
        print('Use device:', device)
        return device

    def _build_model(self):
        if self.args.use_multi_gpu:
            raise ValueError('Health experiments use a single device.')
        if self.args.seq_len % 4:
            raise ValueError('The uploaded FTQNet requires seq_len divisible by its fixed four heads.')
        if self.args.label_len != 0:
            raise ValueError('Health experiments require label_len=0.')
        if self.args.itr != 1:
            raise ValueError('Use separate runs for different random seeds.')
        if self.args.lradj != 'type1':
            raise ValueError('Health protocol requires type1 learning rate.')
        if self.args.train_epochs <= 0 or self.args.patience <= 0:
            raise ValueError('train_epochs and patience must be positive.')
        return Model(self.args).float()

    def _get_data(self, flag):
        return health_data_provider(self.args, flag)

    def _criterion(self):
        return nn.L1Loss() if self.args.loss == 'mae' else nn.MSELoss()

    def _forward(self, x, cycle):
        inputs = x.float().to(self.device)
        index = cycle.to(self.device).int()
        if self.args.use_amp and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return self.model(inputs, index)
        return self.model(inputs, index)

    def _select_targets(self, outputs, y):
        first_channel = -1 if self.args.features == 'MS' else 0
        return (outputs[:, -self.args.pred_len:, first_channel:],
                y[:, -self.args.pred_len:, first_channel:].float().to(self.device))

    def vali(self, loader):
        self.model.eval()
        criterion = self._criterion()
        losses = []
        with torch.no_grad():
            for x, y, _, _, cycle in loader:
                pred, true = self._select_targets(self._forward(x, cycle), y)
                losses.append(criterion(pred, true).item())
        self.model.train()
        value = float(np.mean(losses))
        if not math.isfinite(value):
            raise FloatingPointError('Validation loss is not finite.')
        return value

    def train(self, setting):
        """Train from scratch and restore the validation-best state in memory."""
        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = self._criterion()
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.args.use_amp and self.device.type == 'cuda')

        best_loss = float('inf')
        best_epoch = 0
        bad_epochs = 0
        best_state = None
        history = []
        run_start = time.time()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            losses = []
            for x, y, _, _, cycle in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred, target = self._select_targets(self._forward(x, cycle), y)
                loss = criterion(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Training loss is not finite.')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())

            val_loss = self.vali(val_loader)
            item = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                    'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']}
            history.append(item)
            print('Epoch {} train={:.8f} val={:.8f}'.format(
                item['epoch'], item['train_loss'], val_loss), flush=True)
            if val_loss <= best_loss:
                best_loss, best_epoch, bad_epochs = val_loss, epoch + 1, 0
                best_state = {name: parameter.detach().cpu().clone()
                              for name, parameter in self.model.state_dict().items()}
            else:
                bad_epochs += 1
            if bad_epochs >= self.args.patience:
                print('Early stopping at epoch {} (best epoch {})'.format(
                    epoch + 1, best_epoch), flush=True)
                break
            # Same type1 update as the previously verified health experiment.
            for group in optimizer.param_groups:
                group['lr'] = self.args.learning_rate * (0.5 ** epoch)

        if best_state is None:
            raise RuntimeError('Training produced no valid best state.')
        self.model.load_state_dict(best_state, strict=True)
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        torch.save(best_state, os.path.join(path, 'checkpoint.pth'))
        self.fit_summary = {
            'setting': setting, 'best_epoch': best_epoch,
            'best_val_loss': best_loss, 'history': history,
            'fit_seconds': time.time() - run_start,
            'data_sha256': train_data.manifest()['data_sha256'],
            'test_used_for_selection': False,
        }
        print('Training complete: epochs={} best_epoch={} best_val={:.8f}'.format(
            len(history), best_epoch, best_loss), flush=True)
        return self.model

    def fit_covid_calibration(self, setting, load_best_checkpoint=False):
        """Fit current-horizon rules using validation only; freeze before test."""
        if self.args.data != 'Covid-19' or not getattr(self.args, 'covid_calibrated', False):
            raise ValueError('Explicit calibration is available for calibrated COVID-19 runs only')
        if load_best_checkpoint:
            path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:  # torch<2.0 lacks the weights_only argument
                state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)
        elif not hasattr(self, 'fit_summary'):
            raise RuntimeError('Training must finish before validation calibration')
        # No external parameter file or previous-horizon settings are loaded.
        self._covid_calibration_state = None
        self._covid_calibration_state = _fit_covid_calibration(self)
        return self._covid_calibration_state

    def test(self, setting, test=0):
        """Use the uploaded master's result.txt and test_results PDF format."""
        if test:
            path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            # Load only local checkpoints produced by this project.
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:  # torch<2.0 lacks the weights_only argument
                state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)

        if self.args.data == 'Covid-19':
            if getattr(self, '_covid_calibration_state', None) is None:
                raise RuntimeError('Freeze COVID-19 calibration before reading the test set')
        result_setting = re.sub(r'_fa[^_]+_fc[^_]+_fg[^_]+(?=_seed)', '', setting)
        image_dir = os.path.join('test_results', result_setting)
        os.makedirs(image_dir, exist_ok=True)

        if self.args.data == 'Covid-19':
            metrics = _evaluate_covid_calibrated(self, image_dir)
        else:
            data, loader = self._get_data('test')
            if hasattr(self, 'fit_summary') and data.manifest()['data_sha256'] != self.fit_summary['data_sha256']:
                raise ValueError('Data source changed after training.')
            self.model.eval()
            squared_error = absolute_error = 0.0
            count = windows = 0
            with torch.no_grad():
                for index, (x, y, _, _, cycle) in enumerate(loader):
                    pred, true = self._select_targets(self._forward(x, cycle), y)
                    diff = pred.double() - true.double()
                    squared_error += diff.square().sum().item()
                    absolute_error += diff.abs().sum().item()
                    count += diff.numel()
                    windows += len(x)
                    if index % 20 == 0:
                        inputs = x[0].cpu().numpy()
                        guess = pred[0].cpu().numpy()
                        target = true[0].cpu().numpy()
                        gt = np.concatenate([inputs[:, -1], target[:, -1]])
                        pd = np.concatenate([inputs[:, -1], guess[:, -1]])
                        visual(gt, pd, os.path.join(image_dir, f'{index}.pdf'))
                        import matplotlib.pyplot as plt
                        plt.close('all')
            if windows != len(data) or count == 0:
                raise RuntimeError('Some test windows were skipped.')
            mse, mae = squared_error / count, absolute_error / count
            if not (math.isfinite(mse) and math.isfinite(mae)):
                raise FloatingPointError('Test metric is nonfinite.')
            metrics = {'mse': mse, 'mae': mae, 'windows': windows, 'evaluated_elements': count}

        print('mse:{}, mae:{}'.format(metrics['mse'], metrics['mae']), flush=True)
        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write(result_setting + '  \n')
            f.write('mse:{}, mae:{}'.format(metrics['mse'], metrics['mae']))
            f.write('\n\n')
        return metrics


NONE = {"reference": "none", "weight": 0.0}


REFERENCE_BY_GROUP = {
    "New_cases": "mean28",
    "Cumulative_cases": "last",
    "New_deaths": "mean28",
    "Cumulative_deaths": "last",
}


def _array(value, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 3:
        raise ValueError(f"{name} must be shaped [windows,bars,channels].")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values.")
    return result


def _valid_rule(rule: Mapping) -> tuple[str, float]:
    reference = str(rule.get("reference", "none"))
    weight = float(rule.get("weight", 0.0))
    if reference not in ("none", "last", "mean28"):
        raise ValueError(f"Unsupported reference: {reference}")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Calibration weight must be finite and between 0 and 1.")
    if reference == "none" and weight != 0:
        raise ValueError("Reference 'none' must have zero weight.")
    return reference, weight


def reference_values(history: np.ndarray, reference: str) -> np.ndarray:
    if reference == "last":
        return history[:, -1, :]
    if reference == "mean28":
        count = 28
        return history[:, -min(count, history.shape[1]):, :].mean(axis=1)
    raise ValueError(f"Unknown reference {reference}")


def apply_calibration(pred, history, columns, rules: Mapping[str, Mapping]):
    """Apply nonrecursive forecast-only corrections without mutating predictions."""
    pred = _array(pred, "pred")
    history = _array(history, "history")
    if pred.shape[0] != history.shape[0] or pred.shape[2] != history.shape[2]:
        raise ValueError("pred/history disagree on windows or channels")
    if pred.shape[2] != len(columns):
        raise ValueError("columns do not match prediction channels")
    if history.shape[1] < 1:
        raise ValueError("At least one history point is required")
    positions = group_indices(columns)
    unknown = set(rules) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unexpected calibration groups: {sorted(unknown)}")
    result = pred.copy()
    for group, indices in positions.items():
        if not len(indices):
            continue
        ref_type, weight = _valid_rule(rules.get(group, NONE))
        if weight == 0:
            continue
        # Calculate the reference on the SAME normalized input scale as pred.
        reference = reference_values(history[:, :, indices], ref_type)
        result[:, :, indices] = ((1.0 - weight) * pred[:, :, indices]
                                  + weight * reference[:, None, :])
    if not np.isfinite(result).all():
        raise FloatingPointError("Calibration generated NaN/Inf")
    return result


def errors(pred, target) -> dict:
    p = _array(pred, "pred").astype(np.float64, copy=False)
    y = _array(target, "target").astype(np.float64, copy=False)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("pred/target must match and contain values")
    diff = p - y
    return {"mse": float(np.mean(diff * diff, dtype=np.float64)),
            "mae": float(np.mean(np.abs(diff), dtype=np.float64))}


def relative_gains(baseline: Mapping, candidate: Mapping) -> dict:
    return {key: float((baseline[key] - candidate[key]) / max(baseline[key], 1e-12))
            for key in ("mse", "mae")}


def calibration_validation_split(n: int, horizon: int, min_windows: int = 8) -> tuple[int, int]:
    """Non-overlapping validation target periods for a rolling H-step forecast.

    Keep the previously used 55% tuning split wherever possible (including
    H=24). For longer horizons, shrink the tuning/acceptance windows evenly to
    make room for an H-1 origin embargo. The last tuning target ends strictly
    before the first acceptance target begins. Never consult test data.

    An embargoed holdout is impossible when n < H-1+2*min_windows; fail closed
    rather than leaking labels or silently fitting on the acceptance window.
    """
    if horizon < 1 or min_windows < 1 or n < 1:
        raise ValueError('Invalid rolling-forecast validation dimensions')
    capacity = n - horizon + 1
    if capacity < 2 * min_windows:
        raise ValueError('Not enough validation windows for disjoint target periods: '
                         'windows={}, horizon={}, minimum={}'.format(n, horizon, min_windows))
    tune_stop = max(min_windows, int(n * 0.55))
    gate_start = tune_stop + horizon - 1
    if n - gate_start < min_windows:
        tune_stop = capacity // 2
        gate_start = tune_stop + horizon - 1
    if tune_stop < min_windows or n - gate_start < min_windows:
        raise ValueError('Cannot form disjoint calibration validation periods')
    return tune_stop, gate_start


def _weights_from_current_validation(pred, target, history, reference):
    """Generate candidate mixture weights from this horizon's tuning labels.

    For p + w*(reference-p), MSE has a closed-form optimum. MAE has an
    absolute-deviation optimum at the weighted median of (truth-p)/(ref-p).
    A few interpolations around these optima are compared on the EARLY
    validation portion only, and accepted only on a time-disjoint holdout.
    No fixed legacy calibration strengths are embedded in this code.
    """
    if pred.shape != target.shape or pred.shape[0] != history.shape[0]:
        raise ValueError("Invalid per-horizon calibration tensor shapes")
    anchor = reference_values(history, reference)[:, None, :]
    delta = np.broadcast_to(anchor, pred.shape).astype(np.float64) - pred
    residual = target.astype(np.float64) - pred
    denom = np.sum(delta * delta, dtype=np.float64)
    if denom <= 1e-20:
        return ()
    mse_opt = float(np.clip(np.sum(delta * residual, dtype=np.float64) / denom, 0.0, 1.0))

    positive = np.abs(delta) > 1e-12
    ratios = residual[positive] / delta[positive]
    weights = np.abs(delta[positive])
    order = np.argsort(ratios)
    cumulative = np.cumsum(weights[order])
    mae_opt = float(np.clip(ratios[order[np.searchsorted(cumulative, cumulative[-1] * 0.5)]], 0.0, 1.0))
    proposed = (mse_opt, mae_opt, 0.5 * mse_opt, 0.5 * mae_opt,
                0.5 * (mse_opt + mae_opt), min(1.0, 1.5 * mse_opt),
                min(1.0, 1.5 * mae_opt))
    return tuple(sorted({round(value, 8) for value in proposed if value > 1e-8}))


def tune_on_validation(pred, target, history, columns, min_gain=0.005):
    """Tune on early validation windows; confirm on temporally later windows.

    Prediction targets overlap for adjacent windows; embargo of horizon-1
    windows ensures the selection and acceptance targets do not overlap.
    """
    pred = _array(pred, 'pred')
    target = _array(target, 'target')
    history = _array(history, 'history')
    if pred.shape != target.shape:
        raise ValueError('Validation pred and target mismatch')
    if pred.shape[0] != history.shape[0] or pred.shape[2] != history.shape[2]:
        raise ValueError('Validation history mismatch')
    if not 0 <= min_gain < 1:
        raise ValueError('min_gain must be in [0,1)')
    n, horizon, _ = pred.shape
    tune_stop, gate_start = calibration_validation_split(n, horizon)
    positions = group_indices(columns, strict=len(columns) == 948)
    final_rules = {group: dict(NONE) for group in GROUPS}
    for group, idx in positions.items():
        if not len(idx):
            continue
        cols = [columns[int(i)] for i in idx]
        p = pred[:, :, idx]
        y = target[:, :, idx]
        x = history[:, :, idx]
        base_tune = errors(p[:tune_stop], y[:tune_stop])
        base_gate = errors(p[gate_start:], y[gate_start:])
        best = None
        best_score = 0.0
        ref = REFERENCE_BY_GROUP[group]
        current_weights = _weights_from_current_validation(p[:tune_stop], y[:tune_stop], x[:tune_stop], ref)
        for weight in current_weights:
            rule = {group: {'reference': ref, 'weight': weight}}
            candidate = apply_calibration(p, x, cols, rule)
            trial = errors(candidate[:tune_stop], y[:tune_stop])
            gains = relative_gains(base_tune, trial)
            if min(gains.values()) <= 0:
                continue
            score = gains['mse'] + gains['mae']
            if score > best_score:
                best = (ref, weight, candidate)
                best_score = score
        if best is not None:
            ref, weight, candidate = best
            accepted = errors(candidate[gate_start:], y[gate_start:])
            gate_gains = relative_gains(base_gate, accepted)
            pass_gate = all((g >= min_gain for g in gate_gains.values()))
            if pass_gate:
                final_rules[group] = {'reference': ref, 'weight': weight}
    return final_rules


def _valid_tensor(data, name):
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.size == 0:
        raise ValueError(f"{name} must be nonempty [windows, steps, channels]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} has NaN or infinity")
    return arr


def project_physical(
    group_pred: np.ndarray, history: np.ndarray, columns: Sequence[str],
    normalized_zero: np.ndarray, enabled: Mapping[str, bool],
) -> np.ndarray:
    """Transform predictions without changing feature ordering or model state.

    Uses only the observed history and training-only zero anchors. Forecasts
    remain in the *existing standardized scale*, not original counts.
    """
    group_pred = _valid_tensor(group_pred, "predictions")
    history = _valid_tensor(history, "history")
    if group_pred.shape[0] != history.shape[0] or group_pred.shape[2] != history.shape[2]:
        raise ValueError("Prediction and input windows have incompatible sizes")
    if len(columns) != group_pred.shape[2] or history.shape[1] < 1:
        raise ValueError("Columns or history length do not match forecasts")
    floors = np.asarray(normalized_zero, dtype=group_pred.dtype)
    if floors.shape != (len(columns),) or not np.isfinite(floors).all():
        raise ValueError("Invalid train-only zero anchors")
    if set(enabled) != set(GROUPS) or any(not isinstance(enabled[g], (bool, np.bool_)) for g in GROUPS):
        raise ValueError("Must specify a boolean enable flag for each of the four groups")
    result = group_pred.copy()
    positions = group_indices(columns, strict=(len(columns) == 948))
    for group, indices in positions.items():
        if not len(indices) or not enabled[group]:
            continue
        prediction = group_pred[:, :, indices]
        floor = floors[indices][None, None, :]
        if group.startswith("Cumulative_"):
            # A cumulative forecast below the *observed input* is rarely
            # plausible; avoid enforcing monotonic future steps, since the
            # original data do contain rare retroactive revisions.
            floor = np.maximum(floor, history[:, -1, indices][:, None, :])
        result[:, :, indices] = np.maximum(prediction, floor)
    if not np.isfinite(result).all():
        raise FloatingPointError("Physical projection became nonfinite")
    return result


def choose_projection_on_validation(
    original_pred: np.ndarray, target: np.ndarray, history: np.ndarray,
    columns: Sequence[str], group_rules: Mapping, normalized_zero: np.ndarray,
    min_mae_gain: float = 0.005, min_mse_gain: float = 0.0,
) -> dict:
    """One fixed physical candidate/group; tune / embargo / acceptance.

    Group calibration is selected independently. The physical projection tries just one deterministic
    extra operation per group, chosen from the meaning of each measurement.
    No iterative search on later validation windows or the test set occurs.
    """
    pred = _valid_tensor(original_pred, 'pred')
    target = _valid_tensor(target, 'target')
    history = _valid_tensor(history, 'history')
    if pred.shape != target.shape:
        raise ValueError('Prediction/target shape mismatch')
    if history.shape[0] != pred.shape[0] or history.shape[2] != pred.shape[2]:
        raise ValueError('History incompatible with predictions')
    if not 0 <= min_mae_gain < 1 or not 0 <= min_mse_gain < 1:
        raise ValueError('Invalid improvement thresholds')
    n, horizon, _ = pred.shape
    tune_stop, gate_start = calibration_validation_split(n, horizon)
    group_calibrated = apply_calibration(pred, history, columns, group_rules)
    all_on = {g: True for g in GROUPS}
    candidate = project_physical(group_calibrated, history, columns, normalized_zero, all_on)
    positions = group_indices(columns, strict=len(columns) == 948)
    selected: dict[str, bool] = {g: False for g in GROUPS}
    for group, indices in positions.items():
        if not len(indices):
            continue
        old_tune = errors(group_calibrated[:tune_stop, :, indices], target[:tune_stop, :, indices])
        new_tune = errors(candidate[:tune_stop, :, indices], target[:tune_stop, :, indices])
        old_gate = errors(group_calibrated[gate_start:, :, indices], target[gate_start:, :, indices])
        new_gate = errors(candidate[gate_start:, :, indices], target[gate_start:, :, indices])
        tuning_gain = relative_gains(old_tune, new_tune)
        gate_gain = relative_gains(old_gate, new_gate)
        ok = all((gain['mse'] >= min_mse_gain and gain['mae'] >= min_mae_gain for gain in (tuning_gain, gate_gain)))
        selected[group] = bool(ok)
    return selected


def _predict_covid_split(exp, split):
    data, _ = exp._get_data(split)
    loader = DataLoader(data, batch_size=exp.args.batch_size,
                        shuffle=False, num_workers=0, drop_last=False)
    xs, ys, ps = [], [], []
    exp.model.eval()
    with torch.inference_mode():
        for x, y, _, _, cycle in loader:
            p, t = exp._select_targets(exp._forward(x, cycle), y)
            xs.append(x.cpu().numpy().copy())
            ys.append(t.cpu().numpy().copy())
            ps.append(p.cpu().numpy().copy())
    if not ps or sum(len(p) for p in ps) != len(data):
        raise RuntimeError('Incomplete {} predictions'.format(split))
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(xs), data


def _fit_covid_calibration(exp):
    """Fit and freeze validation-only rules before the first test data access."""
    if exp.args.data != 'Covid-19' or exp.args.pred_len not in (24, 36, 48, 60):
        raise ValueError('COVID-19 calibration requires a supported COVID-19 horizon')
    file = Path(exp.args.root_path) / exp.args.data_path
    vp, vy, vx, val_ds = _predict_covid_split(exp, 'val')
    columns = list(val_ds.columns)
    group_indices(columns, strict=True)
    rules = tune_on_validation(vp, vy, vx, columns, min_gain=0.005)
    zero_floor, source_hash = covid_training_zero_floor(file, columns)
    if hasattr(exp, 'fit_summary') and source_hash != exp.fit_summary['data_sha256']:
        raise RuntimeError('Data file changed during training')
    enabled = choose_projection_on_validation(
        vp, vy, vx, columns, rules, zero_floor,
        min_mae_gain=0.005, min_mse_gain=0.0)
    print('COVID-19 calibration: completed (H={})'.format(exp.args.pred_len), flush=True)
    return {
        'data': 'Covid-19', 'pred_len': exp.args.pred_len,
        'model_identity': id(exp.model),
        'data_sha256': source_hash,
        'columns': tuple(columns), 'rules': rules,
        'zero_floor': zero_floor, 'enabled': enabled,
    }


def _evaluate_covid_calibrated(exp, image_dir):
    """Evaluate test *only after* this experiment's calibration is frozen."""
    state = getattr(exp, '_covid_calibration_state', None)
    if state is None:
        raise RuntimeError('COVID-19 calibration must be frozen on validation before testing')
    if (state['data'] != 'Covid-19' or state['pred_len'] != exp.args.pred_len
            or state['model_identity'] != id(exp.model)):
        raise RuntimeError('Calibration belongs to a different model or forecast horizon')
    file = Path(exp.args.root_path) / exp.args.data_path
    import hashlib
    if state['data_sha256'] != hashlib.sha256(file.read_bytes()).hexdigest():
        raise RuntimeError('COVID-19 CSV changed since the validation calibration')
    columns = list(state['columns'])
    rules, zero_floor, enabled = state['rules'], state['zero_floor'], state['enabled']

    tp, ty, tx, test_ds = _predict_covid_split(exp, 'test')
    if list(test_ds.columns) != columns:
        raise ValueError('COVID-19 channel order changed between val/test')
    final_predictions = project_physical(
        apply_calibration(tp, tx, columns, rules), tx, columns, zero_floor, enabled)
    for ix in range(0, len(tp), 20 * exp.args.batch_size):
        input_values = tx[ix]
        true_values = ty[ix]
        pred_values = final_predictions[ix]
        gt = np.concatenate([input_values[:, -1], true_values[:, -1]])
        pd = np.concatenate([input_values[:, -1], pred_values[:, -1]])
        visual(gt, pd, str(Path(image_dir) / f'{ix // exp.args.batch_size}.pdf'))
        import matplotlib.pyplot as plt
        plt.close('all')
    raw_scores = errors(tp, ty)
    result = errors(final_predictions, ty)
    print('COVID-19 uncalibrated mse:{:.9f}, mae:{:.9f}'.format(
        raw_scores['mse'],raw_scores['mae']), flush=True)
    return {'mse': result['mse'], 'mae': result['mae'],
            'windows': int(tp.shape[0]), 'evaluated_elements': int(tp.size)}
