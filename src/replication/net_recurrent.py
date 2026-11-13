"""半递归网络（论文 §5.1）：δ_k = F_k(I_k, δ_{k−1})。

- 每个时点 k 一套独立权重（论文："these are different for each k"）；
- L = 3：输入 → N1 = N2 = d+15 → 输出 d；ReLU，隐层激活前 batch norm；
- 输入层宽度 f + d（spec §5.1 冻结；论文是 N0 = 2d，差异记 documented deviation）；
- simpler 变体不喂 δ_{k−1}（E1 的架构对照）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.policy.policy import Policy


class StepMLP(nn.Module):
    """单步前馈：Linear → BN → ReLU → Linear → BN → ReLU → Linear。"""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, batch_norm: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)
        self.bn1 = nn.BatchNorm1d(n_hidden) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(n_hidden) if batch_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.out(x)


class SemiRecurrentPolicy(nn.Module, Policy):
    """逐时点独立权重的半递归策略；forward 单步、forward_path 批量。

    recurrent=True：输入 [x_k, δ_{k−1}]（论文做法）；False：只喂 x_k（simpler 对照）。
    """

    def __init__(
        self,
        n_features: int,
        d: int = 2,
        n_steps: int = 30,
        hidden: int | None = None,
        recurrent: bool = True,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d = d
        self.n_steps = n_steps
        self.recurrent = recurrent
        n_hidden = int(hidden if hidden is not None else d + 15)
        n_in = n_features + d if recurrent else n_features
        self.steps = nn.ModuleList(
            [StepMLP(n_in, n_hidden, d, batch_norm=batch_norm) for _ in range(n_steps)]
        )
        self.reset()

    def reset(self) -> None:
        self._prev: Tensor | None = None
        self._k = 0

    def forward(self, x_k: Tensor) -> Tensor:
        """单步：(B, f) → (B, d)；内部维护 δ_{k−1}，reset() 清零。"""
        if self._k >= self.n_steps:
            raise RuntimeError(f"forward 调用超过 n_steps={self.n_steps}")
        net = self.steps[self._k]
        if self.recurrent:
            if self._prev is None:
                self._prev = x_k.new_zeros(x_k.shape[0], self.d)
            inp = torch.cat([x_k, self._prev], dim=-1)
        else:
            inp = x_k
        out = net(inp)
        self._prev = out
        self._k += 1
        return out

    def forward_path(self, x: Tensor) -> Tensor:
        """批量：(B, n, f) → (B, n, d)，不改内部状态（训练/评估用）。"""
        batch, n, _ = x.shape
        prev = x.new_zeros(batch, self.d)
        outputs = []
        for k in range(n):
            net = self.steps[k]
            inp = torch.cat([x[:, k], prev], dim=-1) if self.recurrent else x[:, k]
            prev = net(inp)
            outputs.append(prev)
        return torch.stack(outputs, dim=1)
