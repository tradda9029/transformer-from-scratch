import torch
from attention_lm import AttentionLanguageModel

vocab_size = 10
C = 32
D = 32
block_size = 8

idx_1 = torch.randint(0, vocab_size, (2,4))
idx_2 = torch.randint(0, vocab_size, (1,1))
idx_3 = torch.randint(0, vocab_size, (3,7))

model = AttentionLanguageModel(vocab_size,C,D,block_size)

assert model(idx_1).shape == (2,4,vocab_size)
assert model(idx_2).shape == (1,1,vocab_size)
assert model(idx_3).shape == (3,7,vocab_size)

print("the test is success")
