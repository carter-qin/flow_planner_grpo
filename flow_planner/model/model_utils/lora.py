from __future__ import annotations
import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization standard deviation for LoRA weights (simple replacement for init_fn)
    init_std: float = 0.01
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return (
            self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank
        )


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    def __init__(
        self,
        shape: tuple[int, ...],
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.lora_config = lora_config

        self.w = nn.Parameter(torch.empty(shape))
        nn.init.zeros_(self.w)

        self.w_a = None
        self.w_b = None

        if config := self.lora_config:
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank

            self.w_a = nn.Parameter(torch.empty(shape_a))
            self.w_b = nn.Parameter(torch.empty(shape_b))
            nn.init.normal_(self.w_a, std=self.lora_config.init_std)
            nn.init.normal_(self.w_b, std=self.lora_config.init_std)

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b

    def forward(self, eqn: str, x):
        dtype = x.dtype
        result = torch.einsum(eqn, x, self.w.to(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = torch.einsum(eqn_a, x, self.w_a.to(dtype))
            lora = torch.einsum(eqn_b, lora, self.w_b.to(dtype))

            result = result + lora * config.scaling_value

        return result


class FeedForward(nn.Module):
    """Feed forward module."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.lora_config = lora_config

        self.w_gating = nn.Parameter(torch.empty(2, self.features, self.hidden_dim))
        self.w_linear = nn.Parameter(torch.empty(self.hidden_dim, self.features))

        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            self.w_gating_lora = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(2, self.features, self.lora_config.rank)),
                    nn.Parameter(
                        torch.empty(2, self.lora_config.rank, self.hidden_dim)
                    ),
                ]
            )
            self.w_linear_lora = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(self.hidden_dim, self.lora_config.rank)),
                    nn.Parameter(torch.empty(self.lora_config.rank, self.features)),
                ]
            )
            nn.init.normal_(self.w_gating_lora[0], std=self.lora_config.init_std)
            nn.init.normal_(self.w_gating_lora[1], std=self.lora_config.init_std)
            nn.init.normal_(self.w_linear_lora[0], std=self.lora_config.init_std)
            nn.init.normal_(self.w_linear_lora[1], std=self.lora_config.init_std)

    def _dot(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        lora_weights: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        base = torch.matmul(x, w.to(x.dtype))
        if lora_weights is None:
            return base
        return base + torch.matmul(
            torch.matmul(x, lora_weights[0].to(x.dtype)), lora_weights[1].to(x.dtype)
        )

    def forward(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None
            if self.w_gating_lora is None
            else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.GELU()(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None
            if self.w_gating_lora is None
            else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(
            activations,
            self.w_linear,
            None
            if self.w_linear_lora is None
            else (self.w_linear_lora[0], self.w_linear_lora[1]),
        )
        assert outputs.dtype == dtype
        return outputs


def freeze_non_lora_params(model: nn.Module) -> None:
    """Freeze all parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_trainable_params(model: nn.Module) -> dict[str, int]:
    """Count trainable vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "ratio": f"{trainable / total:.4%}" if total > 0 else "0",
    }
