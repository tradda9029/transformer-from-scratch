import torch
from attention_lm import AttentionLanguageModel

model = AttentionLanguageModel(10,32,32,8)

idx_1 = torch.randint(0,10,(2,8))
idx_2 = torch.randint(0,10,(2,9))

assert model(idx_1).shape == (2,8,10)

try:
    model(idx_2)
except ValueError:
    print("Excepted ValueError occurred")
else:
    raise AssertionError("Expected ValueError was not raised")
