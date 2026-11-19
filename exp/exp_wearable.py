"""Wearable FTQNet experiment with corrected *validation only*.

Unlike the released MixBench loader, validation iterates deterministically
and its loss weights every forecasted element equally. The subject split,
19-channel scaling, model, seed, training batches, and test metric stay fixed.
ILI/COVID and legacy experiments continue to use their own unchanged engine.
"""
import math

import torch

from exp.exp_health import Exp_Health
from data_provider.data_wearable import wearable_data_provider


class Exp_Wearable(Exp_Health):
    def _get_data(self, flag):
        return wearable_data_provider(self.args, flag)

    def vali(self, loader):
        was_training = self.model.training
        self.model.eval()
        sum_loss = 0.0
        n_elements = 0
        n_windows = 0
        try:
            with torch.no_grad():
                for x, y, _, _, cycle in loader:
                    pred, true = self._select_targets(self._forward(x, cycle), y)
                    error = pred.double() - true.double()
                    if self.args.loss == 'mse':
                        sum_loss += error.square().sum().item()
                    elif self.args.loss == 'mae':
                        sum_loss += error.abs().sum().item()
                    else:
                        raise ValueError('Unknown loss for Wearable validation')
                    n_elements += error.numel()
                    n_windows += len(x)
        finally:
            self.model.train(was_training)
        if n_elements == 0 or n_windows != len(loader.dataset):
            raise RuntimeError('Validation omitted windows or had no elements.')
        result = sum_loss / n_elements
        if not math.isfinite(result):
            raise FloatingPointError('Validation loss is NaN or Inf.')
        return result
