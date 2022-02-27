import torch

x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
w = torch.Tensor([1, 2, 3, 4]).requires_grad_()
z = x+y

def hook_fn(grad):
    print(grad)

handle_1 = z.register_hook(hook_fn)

o = w.matmul(z)

def hook_fn2(grad):
    print('grad')

handle_2 = z.register_hook(hook_fn2)
handle_2.remove()

print('=====Start backprop=====')
o.backward()
print('=====End backprop=====')

print('x.grad:', x.grad)
print('y.grad:', y.grad)
print('w.grad:', w.grad)
print('z.grad:', z.grad)