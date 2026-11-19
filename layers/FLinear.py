import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import Tensor
from scipy.special import gamma 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clip_matrix_norm_grouped(matrix, max_norm):
    """按组裁剪分数阶梯度项的 RMS 范数，消除矩阵元素数量对阈值的影响。"""
    num_elements = matrix.shape[1] * matrix.shape[2]
    rms_norms = torch.norm(matrix, p=2, dim=(1, 2), keepdim=True) / math.sqrt(num_elements)
    clip_coef = max_norm / (rms_norms + 1e-8)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    return matrix * clip_coef

class Fractional_Order_Matrix_Differential_Solver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, w, b, alpha, c, groups):
        # 转换超参数为张量以便保存
        alpha_t = torch.tensor(alpha, device=w.device)
        c_t = torch.tensor(c, device=w.device)
        ctx.save_for_backward(input1, w, b, alpha_t, c_t)
        ctx.groups = groups
        
        # 兼容 ND 输入的矩阵乘法：[*, I] @ [I, O] -> [*, O]
        outputs = input1 @ w + b
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        input1, w, b, alpha, c = ctx.saved_tensors
        groups = ctx.groups
        
        # 记录原始形状（比如 [B, L, I]）
        orig_shape = input1.shape
        
        # 1. 计算输入的梯度：[*, O] @ [O, I] -> [*, I]，PyTorch 自动支持高维
        x_grad = grad_outputs @ w.T
        
        # ============ 核心修复：维度折叠 ============
        # 将 input1 展平为 2D: [B * L * ..., I] -> [N, I]
        input1_flat = input1.reshape(-1, orig_shape[-1])
        # 将 grad_outputs 展平为 2D: [B * L * ..., O] -> [N, O]
        grad_outputs_flat = grad_outputs.reshape(-1, grad_outputs.shape[-1])
        # ==========================================
        
        # 2. 调用分组分数阶微分求解器（此时输入必定是标准 2D 矩阵 [N, I]）
        w_grad = Fractional_Order_Matrix_Differential_Solver.Fractional_Order_Matrix_Differential_Linear(
            input1_flat, w, b, alpha, c, groups, grad_outputs_flat
        )
        
        # 3. 计算偏置的梯度：无论多少维，展平后直接在第 0 维求和，缩并为 [O]
        b_grad = grad_outputs_flat.sum(dim=0)
        
        return x_grad, w_grad, b_grad, None, None, None
          
    @staticmethod
    def Fractional_Order_Matrix_Differential_Linear(xs, ws, b, alpha, c, groups, grad_outputs):
        """
        此时输入的 xs 必定是 [N, I]，grad_outputs 必定是 [N, O]
        """
        I, O = ws.shape
        N = xs.shape[0]

        # alpha=1 时显式退化为普通线性层的整数阶权重梯度：X^T @ dL/dY
        # 该分支同时避开 gamma(1-alpha)=gamma(0) 的奇点。
        alpha_val = alpha.item()
        if math.isclose(alpha_val, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return xs.transpose(0, 1) @ grad_outputs
        
        # 限制分组数不超过输出特征数
        G = min(groups, O)
        K = (O + G - 1) // G
        padded_O = G * K
        pad_size = padded_O - O
        
        # 1. 动态零填充
        if pad_size > 0:
            ws_padded = torch.nn.functional.pad(ws, (0, pad_size))
            b_padded = torch.nn.functional.pad(b, (0, pad_size))
            grad_outputs_padded = torch.nn.functional.pad(grad_outputs, (0, pad_size))
        else:
            ws_padded = ws
            b_padded = b
            grad_outputs_padded = grad_outputs
            
        # 2. 重塑张量进行批量分组计算
        ws_reshaped = ws_padded.view(I, G, K)
        b_reshaped = b_padded.view(G, K)
        
        # 3. 使用组内有效输出的均值作为代表，避免固定依赖每组第一列
        # 最后一组可能包含补零列，因此均值只统计原始的有效输出
        valid_mask = (torch.arange(padded_O, device=ws.device) < O).view(G, K).to(ws.dtype)
        valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)
        wf = (ws_reshaped * valid_mask.unsqueeze(0)).sum(dim=2) / valid_counts.unsqueeze(0)  # [I, G]
        wf_t = wf.t()              # [G, I]
        b_rep = (b_reshaped * valid_mask).sum(dim=1) / valid_counts  # [G]
        
        # 4. 常数项 Gamma 计算
        g2 = gamma(2.0 - alpha_val)
        g1 = gamma(1.0 - alpha_val)
        
        # 5. 批量计算 w_main [G, N, I]
        wf_abs = torch.abs(wf_t) + 1e-8
        w_main = xs.unsqueeze(0) * (wf_abs.unsqueeze(1) ** (1 - alpha_val)) / g2
        
        # 6. 批量计算 w_partial [G, N, I]
        xs_wf = xs @ wf            # [N, G]
        term1 = xs_wf.t().unsqueeze(-1) - (xs.unsqueeze(0) * wf_t.unsqueeze(1)) + b_rep.view(G, 1, 1)
        term2 = (torch.sgn(wf_t) * (wf_abs ** -alpha_val) / g1).unsqueeze(1)
        w_partial = term1 * term2
        
        # 7. 分组范数裁剪
        w_partial_clipped = clip_matrix_norm_grouped(w_partial, c.item())
        
        # 8. 得到完整的转置分数阶微分矩阵 [G, I, N]
        w_fractional_grouped = (w_main + w_partial_clipped).transpose(-2, -1)
        
        # 9. 批量矩阵乘法 (BMM): [G, I, N] @ [G, N, K] -> [G, I, K]
        grad_outputs_grouped = grad_outputs_padded.view(-1, G, K).permute(1, 0, 2)  # [G, N, K]
        w_grad_grouped = torch.bmm(w_fractional_grouped, grad_outputs_grouped)
        
        # 10. 重构回原始特征空间并裁剪填充
        w_grad_padded = w_grad_grouped.permute(1, 0, 2).reshape(I, padded_O)
        w_grad = w_grad_padded[:, :O]
        
        return w_grad

class FLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, alpha=0.9, c=1.0, groups=16, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.c = c
        self.groups = groups
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return Fractional_Order_Matrix_Differential_Solver.apply(x, self.weight.T, self.bias, self.alpha, self.c, self.groups)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, groups={self.groups}, bias={self.bias is not None}"
