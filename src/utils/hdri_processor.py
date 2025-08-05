import os
# from httpx import get
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from pathlib import Path
import random
import json

import utils3d

import pyexr
import pandas as pd

# from ..modules.sparse.basic import SparseTensor

class HDRI_Preprocessor:
    def __init__(self, envmap_h, envmap_w):
        self.envmap_h = envmap_h
        self.envmap_w = envmap_w

    def load_hdri(self, hdri_file_path):
        # Load HDRI image
        self.hdri = torch.from_numpy(pyexr.read(hdri_file_path)[..., :3]) # [H, W, 3]
    
    def get_rotate_hdri_cond(self, hdri_rot_roll):
        # Rotate HDRI image
        '''
        hdri_rot: [3] (0, 0, roll)
        '''
        envir_map_ldr, envir_map_hdr, envir_map_hdr_raw, view_dirs_world = self.preprcess_envir_map(self.hdri, np.array([0.,0.,hdri_rot_roll]))
        hdri_cond = torch.cat([envir_map_ldr, envir_map_hdr, view_dirs_world], dim=0).float() # [9, H, W]
        return hdri_cond.unsqueeze(0) # [1, 9, H, W]

    def generate_envir_map_dir(self, hdri_rot):
        lat_step_size = np.pi / self.envmap_h
        lng_step_size = 2 * np.pi / self.envmap_w
        theta, phi = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, self.envmap_h), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, self.envmap_w)], indexing='ij')

        sin_theta = torch.sin(torch.pi / 2 - theta)  # [envH, envW]
        light_area_weight = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [envH, envW]
        assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
        light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]

        # phi = phi + np.pi/2 - np.pi - hdri_rot[2]
        phi = phi - hdri_rot[2] - np.pi/2.
        # phi = phi - np.pi - hdri_rot[2]

        view_dirs = torch.stack([   torch.cos(phi) * torch.cos(theta), 
                                    torch.sin(phi) * torch.cos(theta), 
                                    torch.sin(theta)], dim=-1).view(-1, 3)    # [envH * envW, 3]
        light_area_weight = light_area_weight.reshape(self.envmap_h, self.envmap_w)

        return light_area_weight, view_dirs
            
    def get_light(self, hdr_rgb, incident_dir, flip=False, hdr_weight=None, if_weighted=False):
        # flip the image
        envir_map = hdr_rgb.flip(1) if flip else hdr_rgb

        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        if hdr_weight is not None:
            hdr_weight = self.light_area_weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        incident_dir = incident_dir.clamp(-1, 1)
        theta = torch.arccos(incident_dir[:, 2]).reshape(-1) # top to bottom: 0 to pi
        phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1) # left to right: pi to -pi
        #  x = -1, y = -1 is the left-top pixel of F.grid_sample's input
        query_y = (theta / np.pi) * 2 - 1 # top to bottom: -1-> 1
        query_y = query_y.clamp(-1+10e-8, 1-10e-8)
        query_x = -phi / np.pi # left to right: -1 -> 1
        query_x = query_x.clamp(-1+10e-8, 1-10e-8)

        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float() # [1, 1, N, 2]

        if if_weighted is False or hdr_weight is None:
            light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        else:
            weighted_envir_map = envir_map * hdr_weight
            light_rgbs = F.grid_sample(weighted_envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

            light_rgbs = light_rgbs / hdr_weight.reshape(-1, 1)
                
        return light_rgbs

    def rotate_and_preprcess_envir_map(self, envir_map, c2w, hdri_rot, flip=False, debug=False):
        self.light_area_weight, self.view_dirs = self.generate_envir_map_dir(hdri_rot)

        env_h, env_w = envir_map.shape[0], envir_map.shape[1]
        axis_aligned_transform = torch.from_numpy(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])).float() # Blender's convention

        R_world2cam = c2w[:3, :3].T

        R_final = axis_aligned_transform @ R_world2cam

        view_dirs_world = self.view_dirs @ R_final # [envH * envW, 3]

        rotated_hdr_rgb = self.get_light(envir_map, view_dirs_world, flip=flip)
        rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)

        # hdr_raw
        envir_map_hdr_raw = rotated_hdr_rgb

        # ldr
        envir_map_ldr = rotated_hdr_rgb.clip(0, 1) ** (1/2.2)

        # hdr 
        envir_map_hdr = torch.log1p(10 * rotated_hdr_rgb)
        # rescale hdr to [0, 1]
        envir_map_hdr = (envir_map_hdr / torch.max(envir_map_hdr)).clip(0, 1)

        view_dirs_world = view_dirs_world.reshape(env_h, env_w, 3)

        if debug:
            return envir_map_ldr.permute(2, 0, 1), envir_map_hdr.permute(2, 0, 1), envir_map_hdr_raw.permute(2, 0, 1), view_dirs_world.permute(2, 0, 1) * 0.5 + 0.5
        
        # resize to 256x256
        envir_map_ldr = F.interpolate(envir_map_ldr.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        envir_map_hdr = F.interpolate(envir_map_hdr.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        envir_map_hdr_raw = F.interpolate(envir_map_hdr_raw.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        view_dirs_world = F.interpolate(view_dirs_world.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        return envir_map_ldr * 2. - 1., envir_map_hdr * 2. - 1., envir_map_hdr_raw, view_dirs_world

    def preprcess_envir_map(self, envir_map, hdri_rot, flip=False, debug=False):
        self.light_area_weight, self.view_dirs = self.generate_envir_map_dir(hdri_rot)

        env_h, env_w = envir_map.shape[0], envir_map.shape[1]
        axis_aligned_transform = torch.from_numpy(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])).float() 

        view_dirs_world = self.view_dirs @ axis_aligned_transform # [envH * envW, 3]

        rotated_hdr_rgb = self.get_light(envir_map, view_dirs_world, flip=flip)
        rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)

        # hdr_raw
        envir_map_hdr_raw = rotated_hdr_rgb

        # ldr
        envir_map_ldr = rotated_hdr_rgb.clip(0, 1) ** (1/2.2)

        # hdr 
        envir_map_hdr = torch.log1p(10 * rotated_hdr_rgb)
        # rescale hdr to [0, 1]
        envir_map_hdr = (envir_map_hdr / torch.max(envir_map_hdr)).clip(0, 1)

        view_dirs_world = view_dirs_world.reshape(env_h, env_w, 3)

        if debug:
            return envir_map_ldr.permute(2, 0, 1), envir_map_hdr.permute(2, 0, 1), envir_map_hdr_raw.permute(2, 0, 1), view_dirs_world.permute(2, 0, 1) * 0.5 + 0.5
        
        # resize to 256x256
        envir_map_ldr = F.interpolate(envir_map_ldr.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        envir_map_hdr = F.interpolate(envir_map_hdr.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        envir_map_hdr_raw = F.interpolate(envir_map_hdr_raw.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        view_dirs_world = F.interpolate(view_dirs_world.permute(2, 0, 1).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True).squeeze()
        return envir_map_ldr * 2. - 1., envir_map_hdr * 2. - 1., envir_map_hdr_raw, view_dirs_world
    