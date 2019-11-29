import torch
import torch.nn.functional as F

#Set the random seed for reproducibility
torch.manual_seed(2019)

POLY_DEGREE = 6
W_target = torch.randn(POLY_DEGREE, 1) * 5  #torch.randn: get a set of random value from a standard gaussian distribution
b_target = torch.randn(1) * 5

# print(W_target)
# print(b_target)

def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)  #unsqueeze: insert a dimention at the postion
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)  #.cat: joint two tensor(0 for row & 1 for column)

def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item() #mm: metrix multipulation

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):   #enumerate: enumerate the combination of data and its index
        result += '{:+.2f} x^{} '.format(w, len(W) - i) #format function: {} corresponds with the parameter
    result += '{:+.2f}'.format(b[0])
    return result

def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y

# print(get_batch()[1].size())
fc = torch.nn.Linear(W_target.size(0), 1)   #.size() to output the shape of a tensor
print(fc)

sample_x, sample_y = get_batch(10)
# print(sample_x)
# print(sample_y)

#Initial weight and loss
print("Initial weight: ",fc.weight)    #print weight
fc.zero_grad()      #set the gradient to zero
output = F.smooth_l1_loss(fc(sample_x), sample_y)       #use Huber Loss as loss function
loss = output.item()

#one backprop
output.backward()       #backpropagation
for param in fc.parameters():   #params refer to W and b
    param.data.add_(-0.1 * param.grad.data) #param = param + (-0.1 * param.grad.data)
print("Updated weight: ", fc.weight)
output = F.smooth_l1_loss(fc(sample_x), sample_y)
loss2 = output.item()
print(loss-loss2)

# #iterations
# from itertools import count     #count(intial,step), infinitely iterate
# for batch_idx in count(1):
#     # Get data
#     batch_x, batch_y = get_batch()
#
#     # Reset gradients
#     fc.zero_grad()
#
#     # Forward pass
#     output = F.smooth_l1_loss(fc(batch_x), batch_y)
#     loss = output.item()
#
#     # Backward pass
#     output.backward()
#
#     # Apply gradients
#     for param in fc.parameters():
#         param.data.add_(-0.1 * param.grad.data)
#     print("Updated weight: ", fc.weight)
#     print("loss: ", loss)
#
#     # Stop criterion
#     if loss < 1e-3:
#         break
#
# print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
# print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
# print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))