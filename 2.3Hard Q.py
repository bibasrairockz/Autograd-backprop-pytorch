import torch

##Create a 3x3 matrix A and a 3-element vector x with requires_grad=True. Compute the quadratic form f(x)=(x^T)Ax and calculate the gradient with respect to 𝑥.
A= torch.tensor([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]])
x= torch.tensor([1., 2., 3.], requires_grad= True)
print(A, x)
y= torch.matmul(x, torch.matmul(A,x))
y.backward()
print(y)
print(x.grad, "\n")

# x= torch.tensor([1., 2., 3.], requires_grad= True).reshape(1,3)
# y= torch.tensor([1., 2., 3.], requires_grad= True).reshape(3,1)
# print(x,y)
# print(torch.matmul(y,x))

##Define a function 𝑓(𝑥) = 𝑥1^2 + 3𝑥1𝑥2 + 4𝑥2^2. Compute the Hessian matrix at 𝑥 = [ 1.0 , 2.0 ].
# Create tensor
x = torch.tensor([1.0, 2.0], requires_grad=True)
# Define function
f = x[0]**2 + 3*x[0]*x[1] + 4*x[1]**2
# Compute first gradient
grad_f = torch.autograd.grad(f, x, create_graph=True)[0]
# Compute Hessian
hessian = []
for g in grad_f:
    row = torch.autograd.grad(g, x, retain_graph=True)[0]
    hessian.append(row)
hessian = torch.stack(hessian)
# Print Hessian matrix
print("Hessian matrix:\n", hessian)

##Implement a single step of gradient descent for a logistic regression model. Given input 𝑋 and labels 𝑦, compute the loss using the binary cross-entropy and update the weights.
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=False)
y = torch.tensor([0.0, 1.0, 1.0], requires_grad=False)
w = torch.tensor([0.1, -0.1], requires_grad=True)
loss= 0
lr= 0.01
print("w: ", w)
y_hat= torch.matmul(X, w.T)
y_hat= torch.sigmoid(y_hat)
print("y-hat: ", y_hat)
for i in range (len(y)):
    loss+= y[i]*torch.log(y_hat[i]) + (1-y[i])*torch.log(1-y_hat[i])
print("Loss: ", -(loss/len(y)))
cost= -(loss/len(y))
cost.backward()
print("w grad: ", w.grad)
with torch.no_grad():
    w-= lr*w.grad #using w = w - () will change requires_grad.
print("Final w: ", w, "\n")

##Implement a gradient penalty term for a Wasserstein GAN with gradient penalty (WGAN-GP). Given a real data batch 𝑥 and a fake data batch 𝑥 fake, compute the gradient penalty.
# x_real = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
# x_fake = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
# # print(x_real-x_fake)
# wgan= torch.mean((x_real-x_fake), dim=1).sum()
# print(wgan)
# wgan.backward()
# print(x_real.grad,"\n", x_fake.grad, "\n")

x_real = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
x_fake = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
def D(x):
    return x.sum(dim=1)
epsilon = torch.rand(x_real.size(0), 1)
print("epsilon= ", epsilon)
x_hat = epsilon * x_real + (1 - epsilon) * x_fake
print("x_hat= ", x_hat)
d_hat = D(x_hat)
print("d_hat= ", d_hat)
gradients = torch.autograd.grad(outputs=d_hat, inputs=x_hat,
                                grad_outputs=torch.ones_like(d_hat),
                                create_graph=True, retain_graph=True)[0]
# print("torch.ones_like(d_hat)= ", torch.ones_like(d_hat))
gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
print("gradients.norm(2, dim=1)= ", gradients.norm(2, dim=1))
# print(x_fake.norm(2, dim=1))
print("Gradient penalty:", gradient_penalty.item())

