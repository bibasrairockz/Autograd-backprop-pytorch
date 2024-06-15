import torch

##Create a tensor 𝑥 with requires_grad=True. Define a function f(x)=x^2 . Compute and print the gradient of f with respect to x.
x = torch.tensor([3.0], requires_grad= True)
print(x)
y = x**2
print(y)
y.backward()
print(x.grad, "\n")

##Create a tensor 𝑎 with requires_grad=True. Compute the sum of its elements and calculate the gradient.
a = torch.rand(3, requires_grad= True)
print(a)
y= a.sum()
print(y)
y.backward()
print(a.grad, "\n")

##Create two tensors 𝑥 and 𝑦 with requires_grad=True. Define a function 𝑓(𝑥,𝑦)=𝑥∗𝑦. Compute and print the gradient of 𝑓 with respect to both 𝑥 and 𝑦.
x= torch.randn(3, requires_grad= True)
y= torch.tensor([0., 1., 2.], requires_grad= True)
print(x, y)
z= (x*y)
print(z)
z.sum().backward()
print(x.grad, y.grad)