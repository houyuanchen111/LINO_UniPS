from typing import  Tuple
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.model_utils import *
from ..utils import transformer
from ..utils.ind2sub import *
from ..utils.decompose_tensors import *
from ..utils.utils import *
from einops import rearrange
from ..aggregator import Aggregator
import pywt

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output.float(), **modifier, mode="bilinear", align_corners=self.align_corners).to(torch.bfloat16)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

def _make_scratch(in_shape, out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    """
    
    """
    scratch = nn.Module() 

    
    activation_function = nn.LeakyReLU  

    
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape 

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    
    scratch.layer1_rn = nn.Sequential(
        nn.Conv2d(
            in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        ),
        activation_function() 
    )
    scratch.layer2_rn = nn.Sequential(
        nn.Conv2d(
            in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        ),
        activation_function() 
    )
    scratch.layer3_rn = nn.Sequential(
        nn.Conv2d(
            in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        ),
        activation_function() 
    )
    if len(in_shape) >= 4:
        
        scratch.layer4_rn = nn.Sequential(
            nn.Conv2d(
                in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
            ),
            activation_function() 
        )

    return scratch 


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.LeakyReLU(inplace=False),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


class ImageFeatureExtractor(nn.Module):
    def __init__(self, depth=4,img_size=256, patch_size=8, embed_dim=384):
        super(ImageFeatureExtractor, self).__init__()
        self.aggregator = Aggregator(img_size, patch_size, embed_dim,depth=depth,patch_embed="dinov2_vits14_reg") 
       

    def forward(self, x, nImgArray):
    
        
        feat_list, normal_patch_start_idx = self.aggregator(x) 

        return torch.stack(feat_list,dim=0).permute(1,2,0,3,4).flatten(0,1),normal_patch_start_idx 


class ImageFeatureFusion(nn.Module):
    def __init__(self, 
                 in_channels, 
                 use_efficient_attention=False,
                 out_channels = [256, 512, 1024, 1024],
                 features = 256,
    ):
        super(ImageFeatureFusion, self).__init__()
        _, self.iwt_filter = create_wavelet_filter('db1', 384, 384, torch.bfloat16)
        self.pixel_shuffle = nn.PixelShuffle(2) 
        self.norm = nn.LayerNorm(in_channels)
        self.projects = nn.ModuleList(
                [
                    nn.Sequential(
                        
                        nn.Conv2d(
                            in_channels=in_channels // 4, 
                            out_channels=oc,              
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True 
                        ),
                        nn.LeakyReLU()
        
                    )
                    for oc in out_channels 
                ]
            )
        
        self.resize_layers = nn.ModuleList(
    [
        nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(), 
            nn.ConvTranspose2d(
                in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU() 
        ),
        nn.Sequential( 
             nn.ConvTranspose2d(
                 in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
             ),
             nn.LeakyReLU() 
        ),
        nn.Sequential( 
             nn.Conv2d(
                 in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=1, stride=1, padding=0
             ),
             nn.LeakyReLU() 
        ),
        nn.Sequential(
             nn.Conv2d(
                 in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=2, stride=2, padding=0 
             ),
             nn.LeakyReLU() 
         )
        
        
        
        
        
        
    ]
)


        self.scratch = _make_scratch(
            out_channels,
            features,
            expand=False,
        )

        
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features

        self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 , kernel_size=3, stride=2, padding=1
            )
      

        
    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed
    
    def scratch_forward(self, features) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out
    def forward(self, 
                glc: torch.Tensor, 
                nImgArray: torch.Tensor, 
                chunk_size: int = 6
               ) -> torch.Tensor:

        B = glc.shape[0]

        # 如果不需要分块（总批次大小小于或等于块大小），则直接调用核心实现
        if chunk_size is None or chunk_size >= B:
            return self._forward_impl(glc, nImgArray)

        # 否则，进行分块处理
        all_outputs = []
        # 以 chunk_size 为步长进行循环
        for start_idx in range(0, B, chunk_size):
            # 计算当前块的结束索引
            end_idx = min(start_idx + chunk_size, B)
            
            # 从大的输入张量中切出当前要处理的小块
            glc_chunk = glc[start_idx:end_idx]
            
            # 注意：如果 nImgArray 也与批次相关，也需要进行切片
            # nImgArray_chunk = nImgArray[start_idx:end_idx]
            
            # 调用核心实现函数来处理这个小块
            chunk_output = self._forward_impl(glc_chunk, nImgArray)
            
            all_outputs.append(chunk_output)
            
        # 将所有小块的处理结果，沿着批次维度（dim=0）重新拼接起来
        final_output = torch.cat(all_outputs, dim=0)
        
        return final_output

    def _forward_impl(self, glc: torch.Tensor, nImgArray: torch.Tensor) -> torch.Tensor:
        """
        这是核心实现方法，处理一个数据块（chunk）。
        这里的代码就是您提供的原始 forward 方法的主体。
        """
        self.iwt_filter = self.iwt_filter.to(glc.device)
        B, layer_num, N, C = glc.shape # 这里的 B 现在是 chunk_size
        out = []
        
        for layer in range(layer_num):
            x = glc[:, layer, :, :] # [B, N, C]
            x = self.norm(x) 
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], int(N**0.5), int(N**0.5))) # [B,C,sqrt(N),sqrt(N)]
            
            x = self.pixel_shuffle(x) # [B, C, H, W] -> [B, C/4, H*2, W*2]
            
            x = self.projects[layer](x)
            x = self._apply_pos_embed(x, 256, 256).to(torch.bfloat16)
            x = self.resize_layers[layer](x)
            
            out.append(x)
        
        out = self.scratch_forward(out)
        out = self._apply_pos_embed(out, 256, 256).to(torch.bfloat16) # [B, 256, 64, 64]
        
        return out
    
    




    
class ScaleInvariantSpatialLightImageEncoder(nn.Module): 
    def __init__(self, input_nc, depth=4, use_efficient_attention=False):
        super(ScaleInvariantSpatialLightImageEncoder, self).__init__()
        out_channels = (96, 192, 384, 768)
        self.backbone = ImageFeatureExtractor(depth=depth)
        self.fusion = ImageFeatureFusion(in_channels=1536, use_efficient_attention=use_efficient_attention)
        self.feat_dim = 256
        self.wt_filter, _ = create_wavelet_filter('db1', 3, 3, torch.bfloat16)
        _, self.iwt_filter = create_wavelet_filter('db1', self.feat_dim, self.feat_dim, torch.bfloat16)


    def forward(self, x, nImgArray, canonical_resolution):
        N, C, H, W = x.shape        
        B = N//nImgArray[0]   
        mosaic_scale = H // canonical_resolution
        K = mosaic_scale * mosaic_scale
        self.wt_filter = self.wt_filter.to(x.device)
        self.iwt_filter = self.iwt_filter.to(x.device)

        """ (1a) resizing x to (Hc, Wc)"""
        x_resized = F.interpolate(x.float(), size= (canonical_resolution, canonical_resolution), mode='bilinear', align_corners=True).to(torch.bfloat16) 
        x_resized = x_resized.view(len(nImgArray), int(nImgArray[0]), C, x_resized.shape[2], x_resized.shape[3]) 
        
        
        """ (1b) decomposing x into K x K of (Hc, Wc) non-overlapped blocks (stride)"""           
        x_wt = wavelet_transform(x, self.wt_filter).permute(0, 2, 1, 3, 4) 
    
        x_wt = x_wt.reshape(B,nImgArray[0],K,3,canonical_resolution,canonical_resolution).flatten(1,2).flatten(0,1) 
        x_wt = x_wt.view(len(nImgArray), K * int(nImgArray[0]), C, x_wt.shape[2], x_wt.shape[3])

        """ (2a) feature extraction """
        aggregated_tokens_list, patch_start_idx = self.backbone(x_resized,nImgArray)
        light_tokens_resized = aggregated_tokens_list[:,:,:patch_start_idx - 4,:] 
        light_tokens_resized = rearrange(light_tokens_resized,'(B f) layer num c -> B f layer num c',B = B) 
        x = self.fusion(aggregated_tokens_list[:,:,patch_start_idx:,:], nImgArray)
        f_resized_grid = F.interpolate(x.reshape(N, self.feat_dim, canonical_resolution, canonical_resolution).float() , size= (H, W), mode='bilinear', align_corners=True).to(torch.bfloat16) 
        
        """ (2b) feature extraction (grid) """

        aggregated_tokens_list, patch_start_idx = self.backbone(x_wt,nImgArray)
        light_tokens_wt = aggregated_tokens_list[:,:,:patch_start_idx - 4,:] 
        light_tokens_wt = rearrange(light_tokens_wt,'(B f k) layer num c -> B f k layer num c',B = B, f=nImgArray[0]) 
        light_tokens = torch.cat((light_tokens_resized.unsqueeze(2), light_tokens_wt), dim=2) 


        x = self.fusion(aggregated_tokens_list[:,:,patch_start_idx:,:], nImgArray) 
        x = rearrange(x, '(f k) c h w -> f c k h w ',k=4) 
        x = inverse_wavelet_transform(x, self.iwt_filter) 
       
        """ (3) upsample """
        glc = (f_resized_grid + x)        
    


        return glc,light_tokens


 
class GLC_Upsample(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Upsample, self).__init__()       
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=dim_hidden, ln=True, dim_feedforward = dim_feedforward,use_efficient_attention=False)
       
    def forward(self, x):
        x = self.comm(x)        
        return x

class GLC_Aggregation(nn.Module):
    def __init__(self, input_nc, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False):
        super(GLC_Aggregation, self).__init__()              
        self.aggregation = transformer.AggregationBlock(dim_input = input_nc, num_enc_sab = num_agg_transformer, num_outputs = 1, dim_hidden=dim_aggout, dim_feedforward = dim_feedforward, num_heads=8, ln=True, attention_dropout=0.1, use_efficient_attention=use_efficient_attention)

    def forward(self, x):
        x = self.aggregation(x)      
        return x




class Regressor(nn.Module):
    def __init__(self, input_nc, num_enc_sab=1, use_efficient_attention=False, dim_feedforward=256, output='normal'):
        super(Regressor, self).__init__()     
        
        self.comm = transformer.CommunicationBlock(input_nc, num_enc_sab = num_enc_sab, dim_hidden=input_nc, ln=True, dim_feedforward = dim_feedforward, use_efficient_attention=use_efficient_attention)
        if output == 'normal':   
            self.prediction_normal = PredictionHead(input_nc, 3, confidence=True) 
        self.target = output
     
    def forward(self, x, num_sample_set):
        """Standard forward
        INPUT: img [Num_Pix, F]
        OUTPUT: [Num_Pix, 3]"""  
        if x.shape[0] % num_sample_set == 0:
            x_ = x.reshape(-1, num_sample_set, x.shape[1])
            x_ = self.comm(x_) 
            x = x_.reshape(-1, x.shape[1])
        else:
            ids = list(range(x.shape[0]))
            num_split = len(ids) // num_sample_set
            x_1 = x[:(num_split)*num_sample_set, :].reshape(-1, num_sample_set, x.shape[1])
            x_1 = self.comm(x_1).reshape(-1, x.shape[1])
            x_2 = x[(num_split)*num_sample_set:,:].reshape(1, -1, x.shape[1])
            x_2 = self.comm(x_2).reshape(-1, x.shape[1])
            x = torch.cat([x_1, x_2], dim=0)
        if self.target == 'normal':
            x_n, conf = self.prediction_normal(x.reshape(x.shape[0]//num_sample_set, num_sample_set, -1)) 
            x_ = []
            return x_n, x_, x, conf  
        
  
    
class PredictionHead(nn.Module):
    def __init__(self, dim_input, dim_output, confidence=False):
        
        
        super(PredictionHead, self).__init__()
        modules_regression = []
        modules_regression.append(nn.Linear(dim_input, dim_input//2))
        modules_regression.append(nn.ReLU())
        self.out_layer = nn.Linear(dim_input//2, dim_output)
        if confidence:
            self.confi_layer = nn.Linear(dim_input//2, 1)

        self.regression = nn.Sequential(*modules_regression)

    def forward(self, x):
        h = self.regression(x)
        ret = self.out_layer(h)
        if hasattr(self, 'confi_layer'):
            confidence = self.confi_layer(h) 
        else:
            confidence = torch.zeros_like([ret.shape[0], 1])
        return ret, torch.sigmoid(confidence) 
