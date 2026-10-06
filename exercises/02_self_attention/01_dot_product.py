import torch

a = torch.tensor([
    1,
    2,
    3
])

b = torch.tensor([
    4,
    5,
    6
])

print(a)
print(b)

print(a.shape)
print(b.shape)

elementwise = a * b

print(elementwise)
print(elementwise.shape)

dot_product = elementwise.sum()

print(dot_product)
print(dot_product.shape)

dot_product_torch = torch.dot(a,b)
print(dot_product_torch)
print(dot_product_torch.shape)

print(type(a))
print(type(dot_product))
