import torch
import numpy as np
class MyInvert1(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  @staticmethod
  def forward(ctx, x):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
    """
    ctx.save_for_backward(x)
    return torch.div(1, x)

  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
    """
    x, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x *= -100
    np.savetxt('denominator.csv', np.reshape(x.detach().numpy(),[-1,1]))
    raise ValueError('stop here')
    #print(x)
    return grad_x

class MyInvert2(torch.autograd.Function):
  """
  Second attempt, with grad *= -x
  """
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return torch.div(1, x)

  @staticmethod
  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    grad_x = grad_output.clone()
    #print(np.shape(grad_x.detach().numpy()))
    #np.savetxt('denominator.csv', np.reshape(x.detach().numpy(),[-1,1]))
    #np.savetxt('grad.csv', np.reshape(grad_x.detach().numpy(),[-1,1]))
    #raise ValueError('stop here')
    grad_x *= -x
    #grad_x *= -1*x
    return grad_x

class MyInvert3(torch.autograd.Function):
  """
  Third attempt, with grad *= -x
  """
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return torch.div(1, x)

  @staticmethod
  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x[grad_x < 0] *= -x[grad_x < 0]
    grad_x[grad_x > 0] *= -1
    return grad_x

class MyInvert_original(torch.autograd.Function):
  """
  Third attempt, with grad *= -x
  """

  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return torch.div(1, x)

  @staticmethod
  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    grad_x = grad_output.clone()
    #print("Mean of gradient received by inversion", np.mean(grad_x.detach().numpy(), axis=(0,2)))
    grad_x *= -1/torch.mul(x, x)
    return grad_x

class Mymul_original(torch.autograd.Function):
  """
  Attempt to make multi-input custom auto-grad function
  """
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x,y)
    return torch.mul(x, y)
  @staticmethod
  def backward(ctx, grad):
    x,y, = ctx.saved_tensors
    grad = grad.clone()
    #print("Mean of gradient received by multiple", np.mean(grad.detach().numpy(),axis=(0,2)))
    grad_x = grad * y
    grad_y = grad * x
    return grad_x, grad_y

class Mydiv(torch.autograd.Function):
  """
  Attempt to make multi-input custom auto-grad function
  """
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return torch.div(x, y)
  @staticmethod
  def backward(ctx, grad):
    x,y, = ctx.saved_tensors
    grad = grad.clone()
    grad_x = grad * y
    grad_y = grad * -x
    #grad_x = grad * torch.mul(y, y)
    #grad_y = grad * torch.mul(y, -x)
    #np.savetxt('grad.csv', np.reshape(grad.detach().numpy(),[-1,1]))
    #np.savetxt('numerator.csv', np.reshape(x.detach().numpy(),[-1,1]))
    #np.savetxt('denominator.csv', np.reshape(y.detach().numpy(),[-1,1]))
    #print("shape of grad get by division:", np.shape(grad.detach().numpy()))
    #print("Mean of gradient received by division", np.mean(grad.detach().numpy(), axis=(0,2)))
    #print("Mean of gradient received by division", np.mean(grad.detach().numpy(), axis=(1,2)))
    #print("Mean of gradient received by division", np.mean(grad.detach().numpy(), axis=(0,1)))
    """
    print("gradient received by division", grad.detach().numpy()[0,0,:])
    print("gradient passed to numerator", grad_x.detach().numpy()[0,0,:])
    print("gradient passed to denominator", grad_y.detach().numpy()[0,0,:])
    print("Numerator", x.detach().numpy()[0,0,:])
    print("denominator", y.detach().numpy()[0,0,:])
    raise ValueError("This is intentional stop for track backward gradient")
    """
    return grad_x, grad_y

class Mydiv2(torch.autograd.Function):
  """
  Attempt to make multi-input custom auto-grad function
  """
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x,y)
    return torch.div(x, y)
  @staticmethod
  def backward(ctx, grad):
    x,y, = ctx.saved_tensors
    grad = grad.clone()
    #print("Mean of gradient received by multiple", np.mean(grad.detach().numpy(),axis=(0,2)))
    grad_x = grad
    grad_y = grad * -x / y
    return grad_x, grad_y

class Grad_mon(torch.autograd.Function):
  """
  Monitor gradient custom function
  """
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return x
  @staticmethod
  def backward(ctx, grad):
    x, = ctx.saved_tensors
    grad = grad.clone()
    #print("gradient in grad mon stage", grad.detach().numpy()[0,:])
    #print("Mean of gradient received by multiple", np.mean(grad.detach().numpy(),axis=(0,2)))
    return grad
