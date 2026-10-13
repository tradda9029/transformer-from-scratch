import torch
from attention_lm import AttentionLanguageModel

vocab_size = 10
embedding_dim = 32
head_dim = 32
block_size = 8

model = AttentionLanguageModel(vocab_size,embedding_dim,head_dim,block_size)

idx = torch.tensor([[1,2,3,4]])

idx_new = idx.clone()
idx_new[:, 2:] = torch.tensor([5,6])

assert torch.allclose(model(idx)[:,0:2,:],model(idx_new)[:,0:2,:])

print("the test is success")
