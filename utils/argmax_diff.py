import torch

class ArgMax(torch.autograd.Function):

                @staticmethod
                def forward(ctx, input):
                    idx = torch.argmax(input,keepdim=True)
                    ctx._input_shape = input.shape
                    ctx._input_dtype = input.dtype
                    ctx._input_device = input.device
                    ctx.save_for_backward(idx)
                    return idx.float()

                @staticmethod
                def backward(ctx, grad_output):
                    idx, = ctx.saved_tensors
                    grad_input = torch.zeros(ctx._input_shape, device=ctx._input_device, dtype=ctx._input_dtype)
                    grad_input.scatter_(1, idx[:, None], grad_output.sum())
                    return grad_input
