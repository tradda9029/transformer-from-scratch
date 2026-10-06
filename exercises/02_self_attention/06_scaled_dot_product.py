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

S = Q @ K_T

S_scaled = (Q @ K_T)/math.sqrt(K.shape[-1])

A = torch.softmax(S,dim=-1)

A_scaled = torch.softmax(S_scaled,dim=-1)

Y = A @ V
Y_scaled = A_scaled @ V

print(Y.shape)
print(Y_scaled.shape)

print("A:")
print(A[0])

print("A_scaled:")
print(A_scaled[0])

print(A.max(dim=-1).values)
print(A_scaled.max(dim=-1).values)

print((Y - Y_scaled).abs().max())
