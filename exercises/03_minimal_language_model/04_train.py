import torch
import torch.nn as nn
from tokenizer import vocab_size, encode, decode
from dataset import data, get_batch
from bigram import BigramLanguageModel

batch_size = 8
block_size = 8
max_steps = 1000
learning_rate = 0.01

if __name__ == "__main__":

    model = BigramLanguageModel(vocab_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_steps):
        x,y = get_batch(data, batch_size, block_size)

        logits = model(x)
        logits_flat = logits.reshape(batch_size*block_size,vocab_size)
        targets_flat = y.reshape(batch_size*block_size,)

        loss = loss_fn(logits_flat,targets_flat)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            print(step, loss.item())

    idx = torch.tensor([encode("h")],dtype=torch.long)

    model.eval()

    with torch.no_grad():
        generated = model.generate(idx,50)

    generated_text = decode(generated[0].tolist())


    print("idx.shape:",idx.shape)
    print("generated.shape:",generated.shape)
    print("generated text",generated_text)

    assert idx.shape == (1,1)
    assert generated.shape == (1,51)
    assert generated_text.startswith("h")
    assert len(generated_text) == 51