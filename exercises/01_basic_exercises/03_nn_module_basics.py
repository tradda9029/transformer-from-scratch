import torch
import torch.nn as nn

B = 2
T = 3
C = 4
D = 6

x = torch.randn(B,T,C)

linear = nn.Linear(C,D)

y = linear(x)
print(y.shape)

print(linear.weight.shape)
print(linear.bias.shape)

y_manual = x @ linear.weight.T + linear.bias
print(torch.allclose(y, y_manual))

class TinyModel(nn.Module):
    def __init__(self,C,D):
        super().__init__()
        self.linear = nn.Linear(C,D)

    def forward(self, x):
        return self.linear(x)

model = TinyModel(C,D)
y = model(x)

print(y.shape)

print(model)
print(list(model.parameters()))

for p in model.parameters():
    print(p.shape)

for name, p in model.named_parameters():
    print(name, p.shape)

vocab_size = 10
C = 4

embedding = nn.Embedding(vocab_size, C)

tokens = torch.tensor([
    [1, 3, 5],
    [2, 4, 6]
])

x = embedding(tokens)
print(tokens.shape)
print(x.shape)

print(embedding.weight)
print(embedding.weight.shape)

print(embedding(tokens[0,0]))
print(embedding.weight[1])