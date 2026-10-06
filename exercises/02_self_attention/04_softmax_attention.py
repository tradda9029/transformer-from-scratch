import torch

X = torch.tensor([
    [
        [1,0],
        [0,1],
        [1,1]
    ],
    [
        [2,0],
        [0,2],
        [2,2]
    ]
    ],dtype = torch.float32)

X_T = X.transpose(1,2)

scores = X @ X_T

attention_weight = torch.softmax(scores,dim=2)

print(attention_weight.sum(dim=2))

Y = attention_weight @ X

print(Y)
print(Y.shape)

print(Y[0,0])