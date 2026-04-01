"""Shared helper functions for the demo scripts."""

from __future__ import annotations

from torch import LongTensor


TARGET_VOCAB = {"P": 0, "i": 1, "want": 2, "a": 3, "beer": 4, "S": 5, "E": 6, "U": 7}
INDEX_TO_TOKEN = {index: token for token, index in TARGET_VOCAB.items()}


def make_training_batch(sentences):
    output_batch = []
    target_batch = []
    for source_text, target_text in sentences:
        output_batch.append([TARGET_VOCAB[token] for token in source_text.split()])
        target_batch.append([TARGET_VOCAB[token] for token in target_text.split()])
    return LongTensor(output_batch), LongTensor(target_batch)


def make_inference_batch(sentences):
    output_batch = []
    for sentence in sentences:
        output_batch.append([TARGET_VOCAB[token] for token in sentence.split()])
    return LongTensor(output_batch)
