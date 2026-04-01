"""KV cache backends used by the decoder-only transformer demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .block_manager import BlockAllocator, MetadataEngine, MindsporeAPI


class CacheBackend(Protocol):
    """Protocol implemented by all cache backends."""

    def write(self, values: torch.Tensor) -> None:
        ...

    def transposed_matmul(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class TorchCacheBackend:
    """Simple PyTorch-backed cache used as a reference implementation."""

    num_blocks: int
    num_heads: int
    block_size: int
    head_dim: int

    def __post_init__(self) -> None:
        self.cache = torch.empty(self.num_heads, 0, self.head_dim)

    def write(self, values: torch.Tensor) -> None:
        self.cache = torch.cat([self.cache, values.unsqueeze(1)], dim=1)

    def transposed_matmul(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).matmul(self.cache.transpose(-1, -2)).squeeze(1)

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).matmul(self.cache).squeeze(1)


class MindsporeCacheBackend:
    """MindSpore-backed cache that stores values in fixed-size blocks."""

    def __init__(self, num_blocks: int, num_heads: int, block_size: int, head_dim: int) -> None:
        self.block_allocator = BlockAllocator(num_blocks, block_size, head_dim)
        self.metadata_engine = MetadataEngine(num_blocks, num_heads, block_size)
        self.num_heads = num_heads

    def write(self, values: torch.Tensor) -> None:
        self.block_allocator.alloc(self.metadata_engine.alloc(), values)

    def _stack_cached_values(self):
        return MindsporeAPI.union(
            [self.block_allocator.get(self.metadata_engine.get(head_index)) for head_index in range(self.num_heads)]
        )

    def transposed_matmul(self, x: torch.Tensor) -> torch.Tensor:
        cached_values = self._stack_cached_values()
        return MindsporeAPI.matmul(
            x.unsqueeze(1),
            MindsporeAPI.transpose(cached_values),
        ).squeeze(1)

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        return MindsporeAPI.matmul(x.unsqueeze(1), self._stack_cached_values()).squeeze(1)


class Cache:
    """Facade that exposes a consistent cache API to the transformer."""

    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        block_size: int,
        head_dim: int,
        backend: str = "mindspore",
    ) -> None:
        if backend == "torch":
            self.backend: CacheBackend = TorchCacheBackend(num_blocks, num_heads, block_size, head_dim)
        else:
            self.backend = MindsporeCacheBackend(num_blocks, num_heads, block_size, head_dim)

    def write(self, values: torch.Tensor) -> None:
        self.backend.write(values)

    def transposed_matmul(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.transposed_matmul(x)

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.matmul(x)

    def delete(self) -> None:
        del self.backend
