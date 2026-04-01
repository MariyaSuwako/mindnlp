"""Decoder-only transformer demo with optional KV cache support."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .kv_cache import Cache


def get_attention_padding_mask(sequence: torch.Tensor) -> torch.Tensor:
    """Return an attention mask that hides padding tokens."""
    batch_size, sequence_length = sequence.size()
    padding_mask = sequence.eq(0).unsqueeze(1)
    return padding_mask.expand(batch_size, sequence_length, sequence_length)


def get_attention_subsequent_mask(sequence: torch.Tensor) -> torch.Tensor:
    """Return an upper-triangular mask for autoregressive decoding."""
    attention_shape = [sequence.size(0), sequence.size(1), sequence.size(1)]
    subsequent_mask = np.triu(np.ones(attention_shape), k=1)
    return torch.from_numpy(subsequent_mask).byte()


class PositionalEncoding(nn.Module):
    def __init__(self, target_vocab_size: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(target_vocab_size, d_model)
        position = torch.arange(0, target_vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(2 * d_model) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor, position: int) -> torch.Tensor:
        if position > -1:
            x = x + self.positional_encoding[position][: x.size(0), :]
        else:
            x = x + self.positional_encoding[: x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attention_mask, -1e9)
        attention = nn.Softmax(dim=-1)(scores)
        return torch.matmul(attention, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, num_heads: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(d_model, d_k * num_heads)
        self.key_projection = nn.Linear(d_model, d_k * num_heads)
        self.value_projection = nn.Linear(d_model, d_v * num_heads)
        self.output_projection = nn.Linear(num_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

    def forward(
        self,
        query: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor | None,
        key_cache: Cache | None = None,
        value_cache: Cache | None = None,
    ) -> torch.Tensor:
        residual = query

        if attention_mask is None:
            query_states = self.query_projection(query).view(batch_size, self.num_heads, self.d_k).flatten(0, 1)
            key_states = self.key_projection(query).view(batch_size, self.num_heads, self.d_k).flatten(0, 1)
            value_states = self.value_projection(query).view(batch_size, self.num_heads, self.d_v).flatten(0, 1)

            if key_cache is None or value_cache is None:
                raise ValueError("KV cache must be provided when attention_mask is None.")

            key_cache.write(key_states)
            value_cache.write(value_states)
            attention_scores = key_cache.transposed_matmul(query_states)
            context = value_cache.matmul(nn.Softmax(dim=-1)(attention_scores))
            context = context.contiguous().view(batch_size, -1)
        else:
            query_states = self.query_projection(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            key_states = self.key_projection(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            value_states = self.value_projection(query).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

            expanded_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            context = ScaledDotProductAttention(self.d_k)(
                query_states,
                key_states,
                value_states,
                expanded_mask,
            )
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)

        output = self.output_projection(context)
        return self.layer_norm(output + residual)


class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.output_projection = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = nn.ReLU()(self.input_projection(inputs.transpose(inputs.dim() - 2, inputs.dim() - 1)))
        output = self.output_projection(output).transpose(inputs.dim() - 2, inputs.dim() - 1)
        return self.layer_norm(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, target_length: int, d_model: int, d_ff: int, d_k: int, d_v: int, num_heads: int) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, num_heads)
        self.feed_forward = PositionwiseFeedForwardNetwork(d_model, d_ff)
        self.target_length = target_length

    def forward(
        self,
        decoder_inputs: torch.Tensor,
        batch_size: int,
        attention_mask: torch.Tensor | None,
        block_size: int,
        key_cache: Cache | None,
        value_cache: Cache | None,
    ) -> torch.Tensor:
        if block_size > 0:
            decoder_outputs = self.self_attention(decoder_inputs, batch_size, None, key_cache, value_cache)
        else:
            decoder_outputs = self.self_attention(decoder_inputs, batch_size, attention_mask)
        return self.feed_forward(decoder_outputs)


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size: int,
        target_length: int,
        d_model: int,
        d_ff: int,
        d_k: int,
        d_v: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(target_length + 1, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(target_length, d_model, d_ff, d_k, d_v, num_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        decoder_inputs: torch.Tensor,
        batch_size: int,
        key_caches: list[Cache | None],
        value_caches: list[Cache | None],
        block_size: int = 0,
        position: int = -1,
    ) -> torch.Tensor:
        decoder_outputs = self.target_embedding(decoder_inputs)
        if block_size > 0:
            decoder_outputs = self.position_embedding(decoder_outputs, position)
            attention_mask = None
        else:
            decoder_outputs = self.position_embedding(decoder_outputs.transpose(0, 1), position).transpose(0, 1)
            attention_mask = get_attention_padding_mask(decoder_inputs)

        for layer, key_cache, value_cache in zip(self.layers, key_caches, value_caches):
            decoder_outputs = layer(
                decoder_outputs,
                batch_size,
                attention_mask,
                block_size,
                key_cache,
                value_cache,
            )
        return decoder_outputs


class Transformer(nn.Module):
    def __init__(
        self,
        target_vocab_size: int,
        target_length: int = 10,
        d_model: int = 512,
        d_ff: int = 2048,
        d_k: int = 64,
        d_v: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.decoder = Decoder(target_vocab_size, target_length, d_model, d_ff, d_k, d_v, num_layers, num_heads)
        self.projection = nn.Linear(d_model, target_vocab_size, bias=False)
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model
        self.target_length = target_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

    def _build_caches(self, batch_size: int, sequence_length: int, block_size: int) -> tuple[list[Cache], list[Cache]]:
        num_blocks = math.ceil(batch_size * sequence_length * self.num_heads / block_size)
        key_caches = [
            Cache(num_blocks, batch_size * self.num_heads, block_size, self.d_k) for _ in range(self.num_layers)
        ]
        value_caches = [
            Cache(num_blocks, batch_size * self.num_heads, block_size, self.d_v) for _ in range(self.num_layers)
        ]
        return key_caches, value_caches

    def forward(self, decoder_inputs: torch.Tensor, block_size: int = 0) -> torch.Tensor:
        batch_size = decoder_inputs.size(0)

        if block_size > 0:
            decoder_steps = decoder_inputs.transpose(0, 1)
            key_caches, value_caches = self._build_caches(batch_size, len(decoder_steps), block_size)
            decoder_logits = torch.empty(batch_size, 0, self.target_vocab_size)
            previous_logits: torch.Tensor | None = None

            for position, decoder_step in enumerate(decoder_steps):
                if previous_logits is not None:
                    padding_positions = decoder_step == 0
                    if padding_positions.any():
                        predicted_tokens = previous_logits[:, : self.target_vocab_size - 1].max(1, keepdim=True).indices
                        decoder_step = decoder_step.clone()
                        decoder_step[padding_positions] = predicted_tokens[padding_positions].squeeze(1)

                decoder_output = self.decoder(
                    decoder_step,
                    batch_size,
                    key_caches,
                    value_caches,
                    block_size,
                    position,
                )
                previous_logits = self.projection(decoder_output)
                decoder_logits = torch.cat([decoder_logits, previous_logits.unsqueeze(1)], dim=1)

            for key_cache, value_cache in zip(key_caches, value_caches):
                key_cache.delete()
                value_cache.delete()
        else:
            key_caches = [None] * self.num_layers
            value_caches = [None] * self.num_layers
            decoder_outputs = self.decoder(decoder_inputs, batch_size, key_caches, value_caches)
            decoder_logits = self.projection(decoder_outputs)

        return decoder_logits
