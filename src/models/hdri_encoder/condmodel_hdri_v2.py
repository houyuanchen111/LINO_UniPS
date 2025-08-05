import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from math import sqrt, pi  


# FP16_MODULES = (
#     nn.Conv1d,
#     nn.Conv2d,
#     nn.Conv3d,
#     nn.ConvTranspose1d,
#     nn.ConvTranspose2d,
#     nn.ConvTranspose3d,
#     nn.Linear,
#     sp.SparseConv3d,
#     sp.SparseInverseConv3d,
#     sp.SparseLinear,
# )

# def convert_module_to_f16(l):
#     """
#     Convert primitive modules to float16.
#     """
#     if isinstance(l, FP16_MODULES):
#         for p in l.parameters():
#             p.data = p.data.half()


# def convert_module_to_f32(l):
#     """
#     Convert primitive modules to float32, undoing convert_module_to_f16().
#     """
#     if isinstance(l, FP16_MODULES):
#         for p in l.parameters():
#             p.data = p.data.float()


class SphericalHarmonicsEncoder(nn.Module):  
    """球谐函数编码器，degree=2，输出9通道"""  

    def __init__(self, degree=2):  
        super().__init__()  
        self.degree = degree  

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:  
        """  
        Args:  
            dirs: [B, 3, H, W] 方向向量（单位向量）  
        Returns:  
            sh: [B, (degree+1)^2, H, W] SH编码特征  
        """  
        dirs = F.normalize(dirs, dim=1)  # 保证单位向量  
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]  # [B,H,W]  

        B, H, W = x.shape  
        device = dirs.device  
        sh_list = []  

        # 常数项 l=0  
        sh_list.append(torch.ones_like(x) * (1 / (2 * sqrt(pi))))  

        # l=1 基  
        if self.degree >= 1:  
            sh_list.append(y * sqrt(3 / (4 * pi)))  # m = -1  
            sh_list.append(z * sqrt(3 / (4 * pi)))  # m = 0  
            sh_list.append(x * sqrt(3 / (4 * pi)))  # m = 1  

        # l=2 基  
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
    方向向量编码器：先SH编码，映射为中间通道数，再用卷积编码提取高阶特征  
    输入: [B, 3, H, W]  
    输出: [B, out_dim, H/8, W/8]，和其他编码器输出通道数统一  
    """  

    def __init__(self, out_dim=256, degree=2, base_channels=64, num_blocks=2):  
        """  
        Args:  
            out_dim: 最终输出通道数（如与LDR、LOG编码器一致）  
            degree: SH函数次数，默认2，输出9通道  
            base_channels: SH投影后初始卷积通道数  
            num_blocks: 残差块数量  
        """  
        super().__init__()  
        self.degree = degree  
        self.sh_encoder = SphericalHarmonicsEncoder(degree)  

        # SH9通道映射成 base_channels (如64)  
        self.project = nn.Conv2d((degree + 1) ** 2, base_channels, kernel_size=1, bias=False)  
        self.norm_proj = nn.GroupNorm(16, base_channels)  
        self.act_proj = nn.GELU()  

        # 下采样卷积编码器，类似SimpleEncoder但简化版  
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

        # 可选残差块强化  
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
    """带可学习缩放因子的RMS标准化"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(self.weight.dtype)  # 确保数据类型一致
        norm = torch.mean(x**2, dim=-1, keepdim=True)
        output = self.weight * (x * torch.rsqrt(norm + self.eps))
        return output.to(dtype)  # 恢复原数据类型  


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
    """三路输入通用编码器。下采样8倍至32x32，输出通道dim"""  
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
        # 可扩展加入旋转编码等  

    def forward(self, x, pos=None):  
        B, L, D = x.shape  
        qkv = self.to_qkv(x).view(B, L, 3, self.num_heads, self.head_dim)  
        q, k, v = qkv.unbind(2)  # each [B,L,H,D]  

        # 标准化  
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
        # 三路独立编码器，dim均分给三路  
        route_dim = model_dim // 3  
        self.encoder_ldr = SimpleEncoder(3, route_dim, num_blocks=num_blocks)  
        self.encoder_log = SimpleEncoder(3, route_dim, num_blocks=num_blocks)  
        self.encoder_dir = DirectionEncoder(out_dim=route_dim, num_blocks=num_blocks)  

        self.position_embed = PositionalEncoding2D(model_dim)  

        self.transformer = nn.Sequential(  
            *[TransformerBlock(model_dim, num_heads) for _ in range(num_attn_blocks)]  
        )  
        self.output_dim = model_dim  
        self.token_num = 32 * 32  # 固定token数  

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
            x: [B, 9, 256, 256] 输入张量  
                - 0:3 LDR颜色 (sRGB)  
                - 3:6 Log亮度 (log1p)  
                - 6:9 观察方向 (单位向量)  

        Returns:  
            fused token features: [B, 1024, model_dim]  
        """  
        B = x.shape[0]  
        device = x.device  

        ldr_feat = self.encoder_ldr(x[:, 0:3])    # [B, route_dim, 32, 32]  
        log_feat = self.encoder_log(x[:, 3:6])    # [B, route_dim, 32, 32]  
        dir_feat = self.encoder_dir(x[:, 6:9])    # [B, route_dim, 32, 32]  

        # 通道拼接  
        fused = torch.cat([ldr_feat, log_feat, dir_feat], dim=1)  # [B, model_dim, 32, 32]  

        # 展平并转置成tokens  
        fused = fused.flatten(2).permute(0, 2, 1)  # [B, 1024, model_dim]  

        # 位置编码  
        pos_emb = self.position_embed(B, 32, 32, device)  
        fused = fused + pos_emb  

        # Transformer融合  
        fused = fused.type(self.dtype)
        fused = self.transformer(fused)  # [B, 1024, model_dim]  

        return fused  # b 1024 768


if __name__ == "__main__":  
    model = HDRICondModel()  
    model.train()  # 开启训练模式，确保梯度流通  

    # 构造示例输入 B=2, 9通道, 256x256  
    x = torch.randn(2, 9, 256, 256, requires_grad=True)  

    # 前向计算  
    out = model(x)  
    print(f"Output shape: {out.shape}")  

    # 进行一次简单的loss计算和反向传播，测试梯度计算过程  
    loss = out.sum()  
    loss.backward()  

    print("Forward and backward passes succeeded without inplace errors.")