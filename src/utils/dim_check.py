from __future__ import annotations

from typing import Sequence, Tuple

import torch

def check_shape(
    tensor: torch.Tensor,
    expected: Sequence[int],
    name: str = "tensor",
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} 应为 Tensor，实际为 {type(tensor)}")

    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise ValueError(
            f"{name} 维度数不匹配: 期望 {len(expected)} 维 {tuple(expected)}, "
            f"实际 {len(actual)} 维 {actual}"
        )
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e != -1 and a != e:
            raise ValueError(
                f"{name} 第 {i} 维不匹配: 期望 {e}, 实际 {a} (完整形状 {actual})"
            )

def check_dim(tensor: torch.Tensor, ndim: int, name: str = "tensor") -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} 应为 Tensor，实际为 {type(tensor)}")
    if tensor.dim() != ndim:
        raise ValueError(
            f"{name} 应为 {ndim} 维，实际 {tensor.dim()} 维 (形状 {tuple(tensor.shape)})"
        )

def check_last_dim(tensor: torch.Tensor, dim: int, name: str = "tensor") -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} 应为 Tensor，实际为 {type(tensor)}")
    if tensor.shape[-1] != dim:
        raise ValueError(
            f"{name} 最后一维应为 {dim}, 实际 {tensor.shape[-1]} "
            f"(完整形状 {tuple(tensor.shape)})"
        )

def check_batch_match(
    tensors: Sequence[torch.Tensor],
    names: Sequence[str] | None = None,
    dim: int = 0,
) -> int:
    if not tensors:
        raise ValueError("check_batch_match 传入了空的张量列表")
    names = names or [f"tensor{i}" for i in range(len(tensors))]
    sizes = [t.shape[dim] for t in tensors]
    if len(set(sizes)) != 1:
        detail = ", ".join(f"{n}={s}" for n, s in zip(names, sizes))
        raise ValueError(f"以下张量在第 {dim} 维大小不一致: {detail}")
    return sizes[0]