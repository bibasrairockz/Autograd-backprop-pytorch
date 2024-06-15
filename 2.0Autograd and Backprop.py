import torch

x = torch.arange(3, requires_grad= True, dtype= torch.float) 
print(x, x.std())
y = x + 2
print(y)
z= (y**2)*2
print(z)
# z= z.mean()
# z.backward()
v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float)
z.backward(v)
print(x.grad)

x.requires_grad_(False)
# y= x.detach()
print(x)
with torch.no_grad():
    y = x + 2
    print(y)

weights= torch.ones(4, requires_grad= True)
for i in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()





