"""Simple training and inference smoke test for the refactored project."""

from torch import nn, optim
import time

from src.demo_data import INDEX_TO_TOKEN, TARGET_VOCAB, make_inference_batch, make_training_batch
from src.transformer_decoder_only import Transformer


if __name__ == "__main__":
    sentences = [["S i want a beer", "i want a beer E"], ["S i want a beer", "i want a beer E"]]

    model = Transformer(len(TARGET_VOCAB))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    decoder_inputs, target_batch = make_training_batch(sentences)

    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(decoder_inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch.contiguous().view(-1))
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
        loss.backward()
        optimizer.step()

    #inference_sentences = ["S i want a beer"] * 10
    inference_sentences = ["S P P P P"] * 10
    decoder_inputs = make_inference_batch(inference_sentences)
    begin = time.time()
    for i in range(1):
        predictions = model(decoder_inputs, 16)
    end = time.time()
    print("Time:", end - begin)
    predictions = predictions[:, :, : len(TARGET_VOCAB) - 1].data.max(2, keepdim=True)[1]

    for sentence, output in zip(inference_sentences, predictions):
        print(sentence, "->", [INDEX_TO_TOKEN[index.item()] for index in output.squeeze()])
