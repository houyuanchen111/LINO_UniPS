import numpy as np
import torch
"""array_shape (rows, cols)
ind = row * cols + col
"""
def ind2coords(array_shape, ind):
    row = torch.div(ind, array_shape[1], rounding_mode='trunc')
    col = ind % array_shape[1] # or numpy.mod(ind.astype('int'), array_shape[1])
    coords = torch.zeros((1, 1, len(ind), 2), dtype=torch.float32)
    coords[:, :, :, 1] = 2 * row.to(torch.float32) / array_shape[0] - 1
    coords[:, :, :, 0] = 2 * col.to(torch.float32) / array_shape[1] - 1
    return coords