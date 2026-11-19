import torch
import torch.nn as nn

from layers.FLinear import FLinear


class Model(nn.Module):
    """Complete FTQNet forecasting model."""

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.use_revin = bool(configs.use_revin)

        self.temporalQuery = nn.Parameter(
            torch.zeros(self.cycle_len, self.enc_in),
            requires_grad=True,
        )
        self.channelAggregator = nn.MultiheadAttention(
            embed_dim=self.seq_len,
            num_heads=4,
            batch_first=True,
            dropout=0.5,
        )

        flinear_kwargs = {
            'alpha': configs.flinear_alpha,
            'c': configs.flinear_c,
            'groups': configs.flinear_groups,
        }

        self.input_proj = FLinear(
            self.seq_len,
            configs.d_model,
            **flinear_kwargs,
        )

        self.model = nn.Sequential(
            FLinear(configs.d_model, configs.d_model, **flinear_kwargs),
            nn.GELU(),
            FLinear(configs.d_model, configs.d_model, **flinear_kwargs),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(configs.dropout),
            FLinear(configs.d_model, self.pred_len, **flinear_kwargs),
        )

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # [batch, sequence, channel] -> [batch, channel, sequence]
        x_input = x.permute(0, 2, 1)

        gather_index = (
            cycle_index.view(-1, 1)
            + torch.arange(
                self.seq_len,
                device=cycle_index.device,
            ).view(1, -1)
        ) % self.cycle_len

        query_input = self.temporalQuery[gather_index].permute(0, 2, 1)
        channel_information = self.channelAggregator(
            query=query_input,
            key=x_input,
            value=x_input,
        )[0]

        projected_input = self.input_proj(x_input + channel_information)
        hidden = self.model(projected_input)
        output = self.output_proj(hidden + projected_input).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
