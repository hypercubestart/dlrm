import torch
import numpy as np

def compute_loss(X, X_quantized):
  # normalized l2 loss
  # with torch.no_grad():
  #   return torch.norm(X - X_quantized, p=2) / torch.norm(X, p=2).detach().item()
  return np.linalg.norm(X - X_quantized, ord=2) / np.linalg.norm(X, ord=2)

def Q(X, xmin, xmax):
  # with torch.no_grad():
  #   n = 4 # 4-bit
  #   s = (xmax - xmin) / (2 ** n - 1)
  #   b = xmin

  #   X_quantized = torch.round((torch.clamp(X, xmin, xmax) - b) / s)
  #   return (X_quantized * s + b).detach()

    n = 4 # 4-bit
    s = (xmax - xmin) / (2 ** n - 1)
    b = xmin

    X_quantized = np.round((np.clip(X, xmin, xmax) - b) / s)
    return (X_quantized * s + b)

def quantization_greedy_search(X):
  # with torch.no_grad():
  #   b = 200
  #   r = 0.16
  #   xmin = cur_min = torch.min(X).detach().item()
  #   xmax = cur_max = torch.max(X).detach().item()
  #   loss = compute_loss(X, Q(X, xmin, xmax))
  #   stepsize = (xmax - xmin)/b
  #   min_steps = b * (1 - r) * stepsize
  #   while cur_min + min_steps < cur_max:
  #       loss_l = compute_loss(X, Q(X, cur_min + stepsize, cur_max))
  #       loss_r = compute_loss(X, Q(X, cur_min, cur_max - stepsize))
  #       if loss_l < loss_r:
  #           cur_min = cur_min + stepsize
  #           if loss_l < loss:
  #               loss, xmin = loss_l, cur_min
  #       else:
  #           cur_max = cur_max - stepsize
  #           if loss_r < loss:
  #               loss, xmax = loss_r, cur_max
  #   return Q(X, xmin, xmax).detach()
    # return xmin, xmax
  b = 200
  r = 0.16
  xmin = cur_min = np.min(X)
  xmax = cur_max = np.max(X)
  loss = compute_loss(X, Q(X, xmin, xmax))
  stepsize = (xmax - xmin)/b
  min_steps = b * (1 - r) * stepsize
  while cur_min + min_steps < cur_max:
      loss_l = compute_loss(X, Q(X, cur_min + stepsize, cur_max))
      loss_r = compute_loss(X, Q(X, cur_min, cur_max - stepsize))
      if loss_l < loss_r:
          cur_min = cur_min + stepsize
          if loss_l < loss:
              loss, xmin = loss_l, cur_min
      else:
          cur_max = cur_max - stepsize
          if loss_r < loss:
              loss, xmax = loss_r, cur_max
  return Q(X, xmin, xmax)