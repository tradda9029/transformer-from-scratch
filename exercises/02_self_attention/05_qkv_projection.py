import torch
import torch.nn as nn

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

W_Q = nn.Linear(2, 4, bias=False)
W_K = nn.Linear(2, 4, bias=False)
W_V = nn.Linear(2, 4, bias=False)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

K_T = torch.transpose(K,1,2)

S = Q @ K_T

A = torch.softmax(S,dim=-1)

Y = A @ V

print(Q.shape)
print(K.shape)
print(V.shape)
print(S.shape)
print(A.shape)
print(Y.shape)

print(W_Q.weight.shape)
Q_manual = X @ W_Q.weight.T
print(torch.allclose(Q,Q_manual))