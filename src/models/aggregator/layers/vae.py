import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from . import PatchEmbed
class VAE(nn.Module):
    # VAE代替DINOv2作为encoder的实现
    # Decoder的部分也顺便写了
    def __init__(self):
        super(VAE, self).__init__()  
        self.vae = AutoencoderKL.from_pretrained("/home/lihong/UPS_Lightning/vae/vae/stable-diffusion-3.5-large", subfolder="vae").requires_grad_(False) # vae不训
    def encode(self, x):
        """
        x: [B*f,3,H,W](multi-lihgt) or x: [B,3,H,W](nml)
        """
        z = self.vae.encode(x).latent_dist.sample() # [B*f,16,64,64]
        return z
    def decode(self, latent):
        """
        x: [B,16,64,64] nml的latent
        """
        decode_nml = self.vae.decode(latent).sample
        return decode_nml #