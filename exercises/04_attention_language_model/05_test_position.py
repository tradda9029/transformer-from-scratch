import torch
from attention_lm import AttentionLanguageModel

idx_1 = torch.tensor([[1,2,3]])
idx_2 = torch.tensor([[2,1,3]])

model = AttentionLanguageModel(10,32,32,8)

model.eval()

with torch.no_grad():
    logits_idx_1 = model(idx_1)
    logits_idx_2 = model(idx_2)

assert not torch.allclose(logits_idx_1[:,-1,:],logits_idx_2[:,-1,:])

diff = (logits_idx_1[:, -1, :] - logits_idx_2[:, -1, :]).abs()

print("max difference:", diff.max().item())