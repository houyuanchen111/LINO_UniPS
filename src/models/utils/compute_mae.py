import torch
import numpy as np

def compute_mae(x1, x2, mask = None): # tensor [B, 3, H, W]
    if mask is not None:
        dot = torch.sum(x1 * x2 * mask, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12]).cuda().float()), torch.Tensor([-1.0+1.0e-12]).cuda().float())
        emap = torch.abs(180 * torch.acos(dot)/np.pi) * mask
        mae = torch.sum(emap) / torch.sum(mask)
        return mae, emap
    if mask is None:
        '''
        x1 : [B, H*W, 3]
        x2 : [B, H*W, 3]
        '''
        dot = torch.sum(x1 * x2, dim=1, keepdim=True)
        dot = torch.max(torch.min(dot, torch.Tensor([1.0-1.0e-12]).cuda().float()), torch.Tensor([-1.0+1.0e-12]).cuda().float())
        error = torch.abs(180 * torch.acos(dot)/np.pi) / x1.shape[1] / x1.shape[0]
        return error



def compute_mae_np(x1, x2, mask = None): # numpy array [H, W, 3]
    x1 = np.divide(x1, np.linalg.norm(x1, axis=2, keepdims=True) + 1.0e-12)
    x2 = np.divide(x2, np.linalg.norm(x2, axis=2, keepdims=True) + 1.0e-12)

    if mask is not None:
        dot = np.sum(x1 * x2 * mask[:, :, np.newaxis], axis=-1, keepdims=True)
        dot = np.maximum(np.minimum(dot, np.array([1.0-1.0e-12])), np.array([-1.0+1.0e-12]))
        emap = np.abs(180 * np.arccos(dot)/np.pi) * mask[:, :, np.newaxis]
        mae = np.sum(emap) / np.sum(mask)
        return mae, emap
    if mask is None:
        dot = np.sum(x1 * x2, axis=-1, keepdims=True)
        dot = np.maximum(np.minimum(dot, np.array([1.0-1.0e-12])), np.array([-1.0+1.0e-12]))
        error = np.abs(180 * np.arccos(dot)/np.pi)
        return error