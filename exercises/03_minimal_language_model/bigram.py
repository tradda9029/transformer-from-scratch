import torch
import torch.nn as nn
from tokenizer import vocab_size
from dataset import data, get_batch

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx: torch.Tensor):
        logits = self.embedding(idx)
        return logits

    def generate(self, idx: torch.Tensor, max_new_tokens: int):

        for _ in range(max_new_tokens):
            logits = self(idx)
            last_logits = logits[:,-1,:]

            probs = torch.softmax(last_logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx,idx_next],dim=1)

        return idx





        







if __name__ == "__main__":
    # model = BigramLanguageModel(vocab_size)

    # idx = torch.randint(0,vocab_size,(2,4))
    # logits = model(idx)

    # print(logits.shape)

    # print(model.embedding.weight.shape)

    # assert logits.shape == (2, 4, vocab_size)

    ########

    # model = BigramLanguageModel(vocab_size)

    # batch_size = 2
    # block_size = 4

    # x,y = get_batch(data, batch_size, block_size)

    # logits = model(x)

    # logits_flat = logits.reshape(batch_size*block_size, vocab_size)
    # targets_flat = y.reshape(batch_size*block_size,)

    # loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(logits_flat, targets_flat)

    # print("x shape:", x.shape)
    # print("y shape:", y.shape)
    # print("logits shape:", logits.shape)
    # print("logits_flat shape:", logits_flat.shape)
    # print("targets_flat shape:", targets_flat.shape)

    # print("loss:", loss)
    # print("loss shape:",
    
    #########

    model = BigramLanguageModel(vocab_size)

    idx = torch.randint(0, vocab_size, (2,4))

    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=10)

    print(generated)
    print(generated.shape)

    assert generated.shape == (2,14)
    assert torch.equal(generated[:,:4],idx)