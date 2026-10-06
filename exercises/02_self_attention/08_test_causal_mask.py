import torch
import torch.nn as nn
import math

torch.manual_seed(0)

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

X_new = X.clone()
X_new[0,1] = torch.tensor([100, -100])

W_Q = nn.Linear(2, 4, bias=False)
W_K = nn.Linear(2, 4, bias=False)
W_V = nn.Linear(2, 4, bias=False)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

Q_new = W_Q(X_new)
K_new = W_K(X_new)
V_new = W_V(X_new)

K_T = torch.transpose(K,1,2)
K_T_new = torch.transpose(K_new, 1,2)

S_scaled = (Q @ K_T)/math.sqrt(K.shape[-1])
S_scaled_new = (Q_new @ K_T_new)/math.sqrt(K_new.shape[-1])

M = torch.tril(torch.ones(X.shape[1],X.shape[1]))
M_bool = M.bool()
S_masked = S_scaled.masked_fill(~M_bool,float("-inf"))
S_masked_new = S_scaled_new.masked_fill(~M_bool,float("-inf"))

A_scaled = torch.softmax(S_masked,dim=-1)
A_scaled_new = torch.softmax(S_masked_new,dim=-1)

print(A_scaled)
print(A_scaled_new)

Y = A_scaled @ V
Y_new = A_scaled_new @ V_new

assert torch.allclose(Y[0, 0], Y_new[0, 0])
assert torch.allclose(Y[0, 1], Y_new[0, 1])
assert torch.allclose(Y[0, 2], Y_new[0, 2])
