import torch
from self_attention import SelfAttention

torch.manual_seed(0)

X = torch.randn(2,3,4)

model = SelfAttention(C=4,D=8)

Y = model(X)

assert Y.shape == (2,3,8)

for name, param in model.named_parameters():
    print(name,param.shape)

X_new = X.clone()
X_new[0,2] = torch.tensor([100.,-100.,50.,-50])

Y_new = model(X_new)

assert torch.allclose(
    Y[0,:2],
    Y_new[0,:2]
)

loss = Y[0,0].pow(2).mean()
loss.backward()

for name, param in model.named_parameters():
    print(name, param.grad)
    print(param.grad.abs().sum())
    assert param.grad is not None

a = torch.randn(2,3,4)
b = torch.randn(4,5,4)
c = torch.randn(1,10,4)

a_y = model(a)
b_y = model(b)
c_y = model(c)

print(a_y.shape)
print(b_y.shape)
print(c_y.shape)



