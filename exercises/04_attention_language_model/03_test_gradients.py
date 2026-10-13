import torch
import torch.nn as nn
from attention_lm import AttentionLanguageModel

vocab_size = 10
embedding_dim = 32
head_dim = 32
learning_rate = 0.001
block_size = 8

model = AttentionLanguageModel(vocab_size,embedding_dim,head_dim,block_size)
loss_fn = nn.CrossEntropyLoss()

x = torch.randint(0,vocab_size,(2,4))
y = torch.randint(0,vocab_size,(2,4))

logits = model(x)
logits_flat = logits.reshape(8,vocab_size)
targets_flat = y.reshape(8,)

loss = loss_fn(logits_flat,targets_flat)

loss.backward()

for name, param in model.named_parameters():
    grad = param.grad
    print(name,grad)

    assert grad is not None
    assert torch.count_nonzero(grad).item() > 0

    print(name,grad.shape,grad.norm().item())