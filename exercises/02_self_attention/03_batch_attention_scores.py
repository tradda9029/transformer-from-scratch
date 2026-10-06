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

print(X)
print(X.shape)

print(X.T)
print(X.T.shape)

X_T = X.transpose(1,2)

print(X_T.shape)

scores = X @ X_T

print(scores)
print(scores.shape)