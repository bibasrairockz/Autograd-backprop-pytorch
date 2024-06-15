import torch

##Create a tensor 𝑥 with requires_grad=True. Define a function 𝑓(x)=(𝑥^2+3𝑥+2)^2. Compute and print the gradient of 𝑓 with respect to 𝑥.
x = torch.tensor([1.], requires_grad= True)
print(x)
y= (x**2 + 3*x +2)**2
print(y)
y.backward()
print(x.grad.item(), "\n")

##Create two 2x2 matrices 𝐴 and 𝐵 with requires_grad=True. Compute the matrix product 𝐶 = 𝐴𝐵 and calculate the gradient of the sum of elements of 𝐶 with respect to 𝐴 and 𝐵.
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
print(A, "\n", B)
C= torch.mm(A,B).sum()
print(C)
C.backward()
print(A.grad, B.grad, "\n")

##Define a custom autograd function for the square operation. Use this function in a simple computation and compute the gradient.
def autograd(x):
    y= x**2
    y.backward()
    print(x.grad.item(), "\n")
x= torch.tensor([3.], requires_grad= True)
autograd(x)

# class MySquare(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input ** 2
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = 2 * input * grad_output
#         return grad_input

# # Create tensor
# x = torch.tensor(3.0, requires_grad=True)

# # Use custom square function
# square = MySquare.apply
# y = square(x)

# # Compute gradient
# y.backward()

# # Print gradient
# print("Gradient of y with respect to x:", x.grad.item())