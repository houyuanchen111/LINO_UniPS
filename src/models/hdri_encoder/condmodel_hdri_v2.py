import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from math import sqrt, pi  




class SphericalHarmonicsEncoder(nn.Module):  
    """Spherical Harmonics Encoder, degree=2, output 9 channels"""  

    def __init__(self, degree=2):  
        super().__init__()  
        self.degree = degree  

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:  
        """  
        Args:  
            dirs: [B, 3, H, W] direction vectors (unit vectors)  
        Returns:  
            sh: [B, (degree+1)^2, H, W] SH encoded features  
        """  
        dirs = F.normalize(dirs, dim=1)  # Ensure unit vectors  
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]  # [B,H,W]  

        B, H, W = x.shape  
        device = dirs.device  
        sh_list = []  

        # Constant term l=0  
        sh_list.append(torch.ones_like(x) * (1 / (2 * sqrt(pi))))  

        # l=1 basis  
        if self.degree >= 1:  
            sh_list.append(y * sqrt(3 / (4 * pi)))  # m = -1  
            sh_list.append(z * sqrt(3 / (4 * pi)))  # m = 0  
            sh_list.append(x * sqrt(3 / (4 * pi)))  # m = 1  

        # l=2 basis  
        if self.degree >= 2:  
            sh_list.append(x * y * sqrt(15 / pi) / 2)        # m = -2  
            sh_list.append(y * z * sqrt(15 / (4 * pi)))      # m = -1  
            sh_list.append((2 * z * z - x * x - y * y) * sqrt(5 / (16 * pi)))  # m=0  
            sh_list.append(x * z * sqrt(15 / (4 * pi)))      # m = 1  
            sh_list.append((x * x - y * y) * sqrt(15 / (16 * pi)))  # m=2  

        sh = torch.stack(sh_list, dim=1)  # [B, 9, H, W]  
        return sh  


class DirectionEncoder(nn.Module):  
    """  
    Direction vector encoder: first SH encoding, map to intermediate channels, then use conv encoder to extract higher-order features  
    Input: [B, 3, H, W]  
    Output: [B, out_dim, H/8, W/8], same output channels as other encoders  
    """  

    def __init__(self, out_dim=256, degree=2, base_channels=64, num_blocks=2):  
        """  
        Args:  
            out_dim: final output channels (same as LDR, LOG encoders)  
            degree: SH function degree, default 2, output 9 channels  
            base_channels: initial conv channels after SH projection  
            num_blocks: number of residual blocks  
        """  
        super().__init__()  
        self.degree = degree  
        self.sh_encoder = SphericalHarmonicsEncoder(degree)  

        # Map SH 9 channels to base_channels (e.g., 64)  
        self.project = nn.Conv2d((degree + 1) ** 2, base_channels, kernel_size=1, bias=False)  
        self.norm_proj = nn.GroupNorm(16, base_channels)  
        self.act_proj = nn.GELU()  

        # Downsampling conv encoder, similar to SimpleEncoder but simplified  
        layers = [  
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),  # H/2  
            nn.GroupNorm(16, base_channels),  
            nn.GELU(),  
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False), # H/4  
            nn.GroupNorm(16, base_channels * 2),  
            nn.GELU(),  
            nn.Conv2d(base_channels * 2, out_dim, 3, stride=2, padding=1, bias=False),      # H/8  
            nn.GroupNorm(16, out_dim),  
            nn.GELU()  
        ]  
        self.conv_encoder = nn.Sequential(*layers)  

        # Optional residual blocks for enhancement  
        self.resblocks = nn.Sequential(  
            *[ResidualBlockGN(out_dim) for _ in range(num_blocks)]  
        )  

    def forward(self, dirs):  
        sh_feat = self.sh_encoder(dirs)  
        x = self.project(sh_feat)  
        x = self.norm_proj(x)  
        x = self.act_proj(x)  

        x = self.conv_encoder(x)  
        x = self.resblocks(x)  

        return x  

class RMSNorm(nn.Module):
    """RMS normalization with learnable scaling factor"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(self.weight.dtype)  # Ensure data type consistency
        norm = torch.mean(x**2, dim=-1, keepdim=True)
        output = self.weight * (x * torch.rsqrt(norm + self.eps))
        return output.to(dtype)  # Restore original data type


class ResidualBlockGN(nn.Module):  
    def __init__(self, channels, num_groups=16):  
        super().__init__()  
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)  
        self.norm1 = nn.GroupNorm(num_groups, channels)  
        self.act1 = nn.GELU()  
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)  
        self.norm2 = nn.GroupNorm(num_groups, channels)  

    def forward(self, x):  
        residual = x  
        out = self.conv1(x)  
        out = self.norm1(out)  
        out = self.act1(out)  
        out = self.conv2(out)  
        out = self.norm2(out)  
        out += residual  
        return F.gelu(out)  


class SimpleEncoder(nn.Module):  
    """General encoder for three-way input. Downsample by 8x to 32x32, output channels dim"""  
    def __init__(self, in_channels=3, dim=256, num_blocks=2):  
        super().__init__()  
        mid_dim = dim // 2  
        self.init_conv = nn.Sequential(  
            nn.Conv2d(in_channels, mid_dim, 3, stride=2, padding=1, bias=False),  # 128x128  
            nn.GroupNorm(16, mid_dim),  
            nn.GELU(),  
            nn.Conv2d(mid_dim, dim, 3, stride=2, padding=1, bias=False),         # 64x64  
            nn.GroupNorm(16, dim),  
            nn.GELU(),  
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),             # 32x32  
            nn.GroupNorm(16, dim),  
            nn.GELU(),  
        )  
        self.resblocks = nn.Sequential(  
            *[ResidualBlockGN(dim) for _ in range(num_blocks)]  
        )  

    def forward(self, x):  
        x = self.init_conv(x)  
        x = self.resblocks(x)  
        return x  # [B, dim, 32, 32]  


class PositionalEncoding2D(nn.Module):  
    def __init__(self, dim):  
        super().__init__()  
        self.linear = nn.Linear(2, dim)  

    def forward(self, b, h, w, device):  
        y = torch.linspace(-1, 1, h, device=device)  
        x = torch.linspace(-1, 1, w, device=device)  
        grid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)  # [H, W, 2]  
        grid = grid.reshape(-1, 2).unsqueeze(0).expand(b, -1, -1)       # [B, L, 2]  
        return self.linear(grid)  # [B, L, dim]  


class MultiHeadAttentionRMS(nn.Module):  
    def __init__(self, dim, num_heads, use_rope=False):  
        super().__init__()  
        self.num_heads = num_heads  
        self.head_dim = dim // num_heads  
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  
        self.to_out = nn.Linear(dim, dim)  
        self.use_rope = use_rope  
        self.q_norm = RMSNorm(self.head_dim)  
        self.k_norm = RMSNorm(self.head_dim)  
        # Optionally add rotary encoding, etc.  

    def forward(self, x, pos=None):  
        B, L, D = x.shape  
        qkv = self.to_qkv(x).view(B, L, 3, self.num_heads, self.head_dim)  
        q, k, v = qkv.unbind(2)  # each [B,L,H,D]  

        # Normalization  
        q = self.q_norm(q)  
        k = self.k_norm(k)  

        q = q.transpose(1, 2)  # [B,H,L,D]  
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  

        out = F.scaled_dot_product_attention(q, k, v)  
        out = out.transpose(1, 2).reshape(B, L, D)  
        return self.to_out(out)  


class FeedForwardNet(nn.Module):  
    def __init__(self, dim, mlp_ratio=4.0):  
        super().__init__()  
        hidden = int(dim * mlp_ratio)  
        self.net = nn.Sequential(  
            nn.Linear(dim, hidden),  
            nn.GELU(),  
            nn.Linear(hidden, dim)  
        )  

    def forward(self, x):  
        return self.net(x)  


class TransformerBlock(nn.Module):  
    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_fp16=False):  
        super().__init__()  
        self.norm1 = RMSNorm(dim)  
        self.attn = MultiHeadAttentionRMS(dim, num_heads)  
        self.norm2 = RMSNorm(dim)  
        self.mlp = FeedForwardNet(dim, mlp_ratio)  

    def forward(self, x):  
        h = self.norm1(x)  
        h = self.attn(h)  
        x = x + h  
        h = self.norm2(x)  
        h = self.mlp(h)  
        return x + h  


class HDRICondModel(nn.Module):  
    def __init__(  
        self,  
        model_dim=768,  
        num_heads=8,  
        num_blocks=2,  
        num_attn_blocks=4,  
        use_fp16=False,  
    ):  
        super().__init__()  
        # Three independent encoders, dim equally divided for three routes  
        route_dim = model_dim // 3  
        self.encoder_ldr = SimpleEncoder(3, route_dim, num_blocks=num_blocks)  
        self.encoder_log = SimpleEncoder(3, route_dim, num_blocks=num_blocks)  
        self.encoder_dir = DirectionEncoder(out_dim=route_dim, num_blocks=num_blocks)  

        self.position_embed = PositionalEncoding2D(model_dim)  

        self.transformer = nn.Sequential(  
            *[TransformerBlock(model_dim, num_heads) for _ in range(num_attn_blocks)]  
        )  
        self.output_dim = model_dim  
        self.token_num = 32 * 32  # Fixed token number  

        self.use_fp16 = use_fp16  

        self.initialize_weights()  

        if use_fp16:  
            self.convert_to_fp16()  

        self.dtype = torch.float16 if use_fp16 else torch.float32

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x):  
        """  
        Args:  
            x: [B, 9, 256, 256] input tensor  
                - 0:3 LDR color (sRGB)  
                - 3:6 Log luminance (log1p)  
                - 6:9 view direction (unit vector)  

        Returns:  
            fused token features: [B, 1024, model_dim]  
        """  
        B = x.shape[0]  
        device = x.device  

        ldr_feat = self.encoder_ldr(x[:, 0:3])    # [B, route_dim, 32, 32]  
        log_feat = self.encoder_log(x[:, 3:6])    # [B, route_dim, 32, 32]  
        dir_feat = self.encoder_dir(x[:, 6:9])    # [B, route_dim, 32, 32]  

        # Channel concatenation  
        fused = torch.cat([ldr_feat, log_feat, dir_feat], dim=1)  # [B, model_dim, 32, 32]  

        # Flatten and transpose to tokens  
        fused = fused.flatten(2).permute(0, 2, 1)  # [B, 1024, model_dim]  

        # Positional encoding  
        pos_emb = self.position_embed(B, 32, 32, device)  
        fused = fused + pos_emb  

        # Transformer fusion  
        fused = fused.type(self.dtype)
        fused = self.transformer(fused)  # [B, 1024, model_dim]  

        return fused  # b 1024 768


if __name__ == "__main__":  
    model = HDRICondModel()  
    model.train()  # Enable training mode to ensure gradient flow  

    # Construct example input B=2, 9 channels, 256x256  
    x = torch.randn(2, 9, 256, 256, requires_grad=True)  

    # Forward computation  
    out = model(x)  
    print(f"Output shape: {out.shape}")  

    # Perform a simple loss computation and backward pass to test gradient computation  
    loss = out.sum()  
    loss.backward()  

    print("Forward and backward passes succeeded without inplace errors.")