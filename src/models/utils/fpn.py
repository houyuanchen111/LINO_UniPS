import torch.nn as nn
import torch
import torch.nn.functional as F

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels, fusion_type='sum'):
        super(FeatureFusionModule, self).__init__()
        self.fusion_type = fusion_type
        self.adjusted_features_convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list])
        self.output_conv = nn.Conv2d(out_channels * len(in_channels_list) if fusion_type == 'concat' else out_channels, out_channels, 1)
        
    def forward(self, features):
        h, w = features[0].shape[-2:]
        adjusted_features = []
        
        for feature, conv in zip(features, self.adjusted_features_convs):
            # チャンネル数を調整
            feature = conv(feature)
            # アップサンプリング
            adjusted_feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=True)
            adjusted_features.append(adjusted_feature)
        
        if self.fusion_type == 'concat':
            fused_feature = torch.cat(adjusted_features, dim=1)
        elif self.fusion_type == 'sum':
            fused_feature = sum(adjusted_features)
        else:
            raise ValueError(f"Invalid fusion_type: {self.fusion_type}")
        
        fused_feature = self.output_conv(fused_feature)
        return fused_feature


# # 特徴マップのリスト
# feature_maps = [
#     torch.randn(10, 32, 64, 64),
#     torch.randn(10, 64, 32, 32),
#     torch.randn(10, 128, 16, 16),
#     torch.randn(10, 256, 8, 8),
# ]

# # モデルの構築
# in_channels_list = [32, 64, 128, 256]  # 各レベルのチャンネル数
# out_channels = 64  # 融合後のチャンネル数を64に指定
# fusion_type = 'sum'  # 'sum'または'concat'を指定

# feature_fusion_module = FeatureFusionModule(in_channels_list, out_channels, fusion_type=fusion_type)

# # 融合
# fused_feature = feature_fusion_module(feature_maps)

# print(fused_feature.shape)  # 融合後の特徴マップのサイズとチャンネル数を確認
