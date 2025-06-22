"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def divide_tensor_spatial(x, block_size=256, method='tile_stride'):
    assert x.dim() == 4, "Input tensor must have 4 dimensions [B, C, H, W]"
    B, C, H, W = x.shape    
    assert H == W, "Height and Width must be equal"
    assert H % block_size == 0 and W % block_size ==0, "The tensor size cannot be divided by the block size"
    mosaic_scale = H // block_size
    
    if method == 'tile_stride':
        """ decomposing x into K x K of (Hc, Wc) non-overlapped blocks (grid)"""           
        
        K = mosaic_scale * mosaic_scale
        fold_params_grid = dict(kernel_size=(mosaic_scale, mosaic_scale), stride=(mosaic_scale, mosaic_scale), padding=(0,0), dilation=(1,1))
        unfold_grid = nn.Unfold(**fold_params_grid)   
        tensor_grids = unfold_grid(x) # (B, C * K, Hm * Hm)
        tensor_grids = tensor_grids.reshape(B, C, K, block_size, block_size).permute(0, 2, 1, 3, 4) # (B, K, C, Hm, Hm)
        return tensor_grids
    
    if method == 'tile_block':    
        tensor_blocks = x.view(B, C, mosaic_scale, block_size, mosaic_scale, block_size)
        tensor_blocks = tensor_blocks.permute(0, 2, 4, 1, 3, 5) # (B, mc, mc, C, Hm, Wm)
        tensor_blocks = tensor_blocks.contiguous().view(B, mosaic_scale**2, C, block_size, block_size) ## (B, K, C, Hm, Hm)
        return tensor_blocks
    
    return -1

def merge_tensor_spatial(x, method='tile_stride'):
    
    K, N, feat_dim, Hm, Wm = x.shape
    mosaic_scale = int(math.sqrt(K))

    if method == 'tile_stride':
        x = x.reshape(K, N, feat_dim, -1)
        fold_params_grid = dict(kernel_size=(mosaic_scale, mosaic_scale), stride=(mosaic_scale, mosaic_scale), padding=(0,0), dilation=(1,1))
        fold_grid = nn.Fold(output_size=(Hm * mosaic_scale, Wm * mosaic_scale), **fold_params_grid) #  downsample based on the encoder     
        x = x.permute(1, 2, 0, 3).reshape(N, feat_dim * K, -1) 
        x = fold_grid(x)
        return x

    if method == 'tile_block':
        x = x.permute(1, 0, 2, 3, 4).reshape(N, mosaic_scale, mosaic_scale, feat_dim, Hm, Wm)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(N, feat_dim, mosaic_scale * Hm, mosaic_scale * Wm)
        return x

def divide_overlapping_patches(input_tensor, patch_size, margin):
    B, C, W, _ = input_tensor.shape
    stride = patch_size - margin
    padded_W = ((W - patch_size + stride - 1) // stride) * stride + patch_size
    pad = padded_W - W

    padded_tensor = F.pad(input_tensor, (0, pad, 0, pad), mode='constant', value=0)

    patches = F.unfold(padded_tensor, kernel_size=patch_size, stride=stride)
    patches = patches.view(B, C, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)

    return patches

def merge_overlappnig_patches(patches, patch_size, margin, original_size):
    B, _, C, _, _ = patches.shape
    stride = patch_size - margin
    W = original_size[2]

    patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(B, C * patch_size * patch_size, -1)
    output = F.fold(patches, (W, W), kernel_size=patch_size, stride=stride)  

    weight = torch.ones(patches.size()).to(patches.device)
    weight = F.fold(weight, (W, W), kernel_size=patch_size, stride=stride)

    output = output / weight
    return output

