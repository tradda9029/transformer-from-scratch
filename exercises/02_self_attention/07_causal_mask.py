import torch
import torch.nn as nn
import math

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

S_scaled = (Q @ K_T)/math.sqrt(K.shape[-1])

M = torch.tril(torch.ones(X.shape[1],X.shape[1]))
M_bool = M.bool()
S_masked = S_scaled.masked_fill(~M_bool,float("-inf"))

A_scaled = torch.softmax(S_masked,dim=-1)

print(A_scaled)

Y_scaled = A_scaled @ V

print(Y_scaled.shape)