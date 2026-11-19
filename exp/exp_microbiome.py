"""Paper-protocol fit/selection for the isolated three microbial datasets.

All other datasets keep using their original experiment classes unchanged.
MicroSTNet author code: Adam (fixed 1e-3), L1, 200 epochs without early
stopping, select minimum full-validation MAE, with best checkpoint saved.
The inherited test() preserves all original FTQNet output paths and formats.
"""
import os
import time

import torch
from torch import nn

from exp.exp_main import Exp_Main


class Exp_Microbiome(Exp_Main):
    def _select_criterion(self):
        return nn.L1Loss()

    def vali(self, vali_data, vali_loader, criterion):
        # The official implementation takes the mean across ALL validation
        # entries, rather than averaging per-batch means (last batch is short).
        absolute_error_sum = 0.0
        num_values = 0
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, _, batch_cycle in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                pred = self.model(batch_x, batch_cycle)
                target = batch_y[:, -self.args.pred_len:, :]
                absolute_error_sum += torch.abs(pred - target).sum().item()
                num_values += target.numel()
        self.model.train()
        if num_values == 0:
            raise ValueError('Empty validation set.')
        return absolute_error_sum / num_values

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, 'checkpoint.pth')

        # Adam's real initial learning rate remains 1e-3. Do not construct
        # OneCycleLR here: its constructor silently changes Adam's initial LR.
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        best_mae = float('inf')
        best_epoch = -1
        for epoch in range(self.args.train_epochs):
            epoch_start = time.time()
            self.model.train()
            running_loss, num_values = 0.0, 0
            for batch_x, batch_y, _, _, batch_cycle in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:, -self.args.pred_len:, :]
                batch_cycle = batch_cycle.int().to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', enabled=bool(self.args.use_amp and self.args.use_gpu)):
                    output = self.model(batch_x, batch_cycle)
                    loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_y.numel()
                num_values += batch_y.numel()

            training_mae = running_loss / max(num_values, 1)
            val_mae = self.vali(vali_data, vali_loader, criterion)
            if val_mae < best_mae:
                best_mae = val_mae
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Validation MAE improved; saved checkpoint.')
            print('Epoch: {}/{} | train MAE {:.7f} | val MAE {:.7f} | best val MAE {:.7f} | {:.2f}s'.format(
                epoch + 1, self.args.train_epochs, training_mae, val_mae,
                best_mae, time.time() - epoch_start))

        print('Paper-aligned fit finished: best epoch {} validation MAE {:.7f}'.format(best_epoch, best_mae))
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        return self.model
