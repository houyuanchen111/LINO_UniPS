import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
import os
from . import PatchEmbed
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()  
        model_id = os.getenv("VAE_MODEL_ID", "stabilityai/stable-diffusion-3.5-large")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").requires_grad_(False)
    def encode(self, x):
        """
        x: [B*f,3,H,W](multi-lihgt) or x: [B,3,H,W](nml)
        """
        z = self.vae.encode(x).latent_dist.sample() # [B*f,16,64,64]
        return z
    def decode(self, latent):
        """
        latent: [B,16,64,64]
        """
        decode_nml = self.vae.decode(latent).sample
        return decode_nml #