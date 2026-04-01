"""Memory block helpers used by the KV cache implementation."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from mindspore import Parameter, float32, tensor
from mindspore.common.initializer import Zero, initializer
from mindspore.ops import ScatterNdUpdate, gather, stack


class BlockAllocator:
    """Stores cache data in fixed-size MindSpore blocks."""

    def __init__(self, num_blocks: int, block_size: int, head_dim: int) -> None:
        self.num_blocks = num_blocks
        self.pool = [
            Parameter(initializer(Zero(), [block_size, head_dim], float32))
            for _ in range(num_blocks)
        ]

    def alloc(self, indices: Sequence[Sequence[int]], values: Iterable[torch.Tensor]) -> None:
        """Write one token per head into the allocated block positions."""
        for index, value in zip(indices, values):
            block_index, offset = index
            self.pool[block_index] = ScatterNdUpdate()(
                self.pool[block_index],
                tensor([[offset]]),
                tensor(value.detach().numpy()).unsqueeze(0),
            )

    def get(self, indices: Sequence[Sequence[int]]):
        """Read back a stacked tensor for the provided block coordinates."""
        values = []
        for block_index, offset in indices:
            values.append(gather(self.pool[block_index], tensor(offset), 0))
        return stack(values)


class MetadataEngine:
    """Tracks token-to-block mappings for all attention heads."""

    def __init__(self, num_blocks: int, num_heads: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.block_size = block_size
        self.block_table: List[List[List[int]]] = []
        self.num_token = 0
        self.block_cursor = 0
        self.offset_cursor = 0

    def alloc(self) -> List[List[int]]:
        """Allocate one position for each head for the next token."""
        token_indices: List[List[int]] = []
        self.num_token += 1

        for _ in range(self.num_heads):
            if self.block_cursor == self.num_blocks:
                raise ValueError("Out of memory in block allocator.")

            token_indices.append([self.block_cursor, self.offset_cursor])
            self.offset_cursor += 1

            if self.offset_cursor == self.block_size:
                self.block_cursor += 1
                self.offset_cursor = 0

        self.block_table.append(token_indices)
        return token_indices

    def get(self, head_index: int) -> List[List[int]]:
        """Return all stored positions for a single attention head."""
        return [self.block_table[token_index][head_index] for token_index in range(self.num_token)]


class MindsporeAPI:
    """Thin wrappers that keep tensor conversion logic in one place."""

    @staticmethod
    def union(values):
        return stack(values)

    @staticmethod
    def transpose(values):
        return values.transpose(-1, -2)

    @staticmethod
    def matmul(x: torch.Tensor, values) -> torch.Tensor:
        return torch.from_numpy(tensor(x.detach().numpy()).matmul(values).asnumpy())
