import torch

X = torch.tensor([
    [1,0],
    [0,1],
    [1,1]
],dtype = torch.float32)

print(X)
print(X.shape)
print(X.dtype)

X_T = X.T

print(X_T)
print(X_T.shape)

scores = X @ X_T

print(scores)
print(scores.shape)






