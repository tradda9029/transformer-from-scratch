import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):

    def __init__(self,C,D):
        super().__init__()

        self.q_proj = nn.Linear(C,D,bias=False)
        self.k_proj = nn.Linear(C,D,bias=False)
        self.v_proj = nn.Linear(C,D,bias=False)

    def forward(self,X):
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        K_T = torch.transpose(K,1,2)

        S = (Q @ K_T) / math.sqrt(K.shape[-1])

        M = torch.tril(
            torch.ones(X.shape[1],X.shape[1],dtype=torch.bool,device=X.device)
            )

        S_masked = S.masked_fill(~M,float("-inf"))

        A = torch.softmax(S_masked,dim=-1)

        Y = A @ V

        return Y