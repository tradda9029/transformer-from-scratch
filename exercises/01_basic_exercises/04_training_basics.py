import torch
import torch.nn as nn

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

target = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])

model = nn.Linear(1,1)

loss_fn = nn.MSELoss()

prediction = model(x)
loss = loss_fn(prediction, target)

print(loss.shape)

print(model.weight)
print(model.bias)

loss.backward()

print(model.weight.grad)
print(model.bias.grad)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("before", model.weight.item())

optimizer.step()

print("after", model.weight.item())


for i in range(1000):
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, target)
    loss.backward()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()


print(model.weight.item())
print(model.bias.item())