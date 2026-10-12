import torch
from src.self_attention import SelfAttention

torch.manual_seed(0)

X = torch.randn(2,3,4)

model = SelfAttention(C=4,D=8)

Y = model(X)

assert Y.shape == (2,3,8)

for name, param in model.named_parameters():
    print(name,param.shape)

X_new_1 = X.clone()
X_new_1[0,2] = torch.tensor([100.,-100.,50.,-50])

Y_new_1 = model(X_new_1)

assert torch.allclose(
    Y[0,0:2],
    Y_new_1[0,0:2]
)

assert torch.allclose(
    Y[1,0:2],
    Y_new_1[1,0:2]
)

X_new_2 = X.clone()
X_new_2[0,1] = torch.tensor([100.,-100.,50.,-50])

Y_new_2 = model(X_new_2)

assert torch.allclose(
    Y[0,0:1],
    Y_new_2[0,0:1]
)

assert torch.allclose(
    Y[1,0:1],
    Y_new_2[1,0:1]
)

loss = Y.pow(2).mean()
loss.backward()

for name, param in model.named_parameters():
    print(name, param.grad)
    print(param.grad.abs().sum())
    assert param.grad is not None

model.zero_grad()
Y_new = model(X)
loss_1 = Y_new[0,0].pow(2).mean()
loss_1.backward()
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

assert a_y.shape == (2,3,8)
assert b_y.shape == (4,5,8)
assert c_y.shape == (1,10,8)



