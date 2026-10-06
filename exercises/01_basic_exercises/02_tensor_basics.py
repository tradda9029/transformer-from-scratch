import torch

B=2
T=3
C=4

x = torch.randn(B,T,C)

print(x)
print(x.shape)
print(x.ndim)
print(x.dtype)
print(x.device)

print(x[0].shape)
print(x[0,0].shape)
print(x[:,0,:].shape)
print(x[:,:,0].shape)

D = 6
W = torch.randn(C,D)

y = torch.matmul(x,W)

print(W.shape)
print(y.shape)

y2 = x @ W

print(y2.shape)

z = torch.randn(2,3,4)
z2 = z.transpose(1,2)
print(z2.shape)
z3 = z.reshape(2,6,2)
print(z3.shape)

z4 = z.reshape(2,-1)
print(z4.shape)
z5 = z.reshape(-1,4)
print(z5.shape)