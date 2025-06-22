import torch.nn as nn
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def imshow(img, mask = None, vmax = None, axis = None):
    if mask is None:
        img = img.data.cpu().numpy()[0,:,:,:]
    else:
        img = masking(img,mask).data.cpu().numpy()[0,:,:,:]
    c = img.shape[0]
    h = img.shape[1]
    w = img.shape[2]
    if c == 3:
        img = np.reshape(img, (3,h,w)).transpose(1,2,0)
    if c == 1:
        img = img[0,:,:]
    # plt.figure(figsize = (8,8))
    if vmax is None:
        if axis is None:
            plt.imshow(img)
        else:
            axis.imshow(img)
    else:
        if axis is None:
            plt.imshow(img, vmax = vmax)
        else:
            axis.imshow(img, vmax = vmax)

def nmlshow(nml, mask = None, axis = None):
    if mask is None:
        nml = nml.data.cpu().numpy()[0,:,:,:]
    else:
        nml = masking(nml,mask).data.cpu().numpy()[0,:,:,:]
    nml = np.transpose(nml, (1,2,0))
    # plt.figure(figsize = (8,8))

    if axis is None:
        plt.imshow(-0.5 * nml + 0.5)
    else:
        axis.imshow(-0.5 * nml + 0.5)
