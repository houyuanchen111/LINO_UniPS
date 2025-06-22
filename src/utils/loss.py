import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class MultiScaleDerivativeLoss(nn.Module):
    def __init__(self, operator='scharr', p=1, reduction='mean', normalize_input=False, num_scales=4):
        """
        operator: 'scharr' (一阶) or 'laplace' (二阶)
        p: 1 for L1, 2 for L2
        reduction: 'mean' or 'sum'
        normalize_input: whether to normalize input vectors (for normals)
        num_scales: number of scales in the pyramid (e.g., 4 = 原图, 1/2, 1/4, 1/8)
        """
        super().__init__()
        assert operator in ['scharr', 'laplace']
        assert p in [1, 2]
        assert reduction in ['mean', 'sum']
        assert num_scales >= 1

        self.operator = operator
        self.p = p
        self.reduction = reduction
        self.normalize_input = normalize_input
        self.num_scales = num_scales

    def forward(self, pred, gt):
        """
        pred, gt: [B, C, H, W] tensors
        """
        pred_pyramid = self._build_pyramid(pred)
        gt_pyramid = self._build_pyramid(gt)

        total_loss = 0.0

        for pred_i, gt_i in zip(pred_pyramid, gt_pyramid):
            if self.normalize_input:
                pred_i = F.normalize(pred_i, dim=1)
                gt_i = F.normalize(gt_i, dim=1)

            grad_pred = self._compute_gradient(pred_i)
            grad_gt = self._compute_gradient(gt_i)

            diff = grad_pred - grad_gt
            if self.p == 1:
                diff = torch.abs(diff)
            else:
                diff = diff ** 2

            if self.reduction == 'mean':
                total_loss += diff.mean()
            else:
                total_loss += diff.sum()

        return total_loss / self.num_scales

    def _build_pyramid(self, img):
        """Construct a multi-scale pyramid from input image"""
        pyramid = [img]
        for i in range(1, self.num_scales):
            scale = 0.5 ** i
            img = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False, recompute_scale_factor=True,antialias=True)
            pyramid.append(img)
        return pyramid

    def _compute_gradient(self, img):
        B, C, H, W = img.shape
        device = img.device

        if self.operator == 'scharr':
            kernel_x = torch.tensor([[[-3., 0., 3.],
                                      [-10., 0., 10.],
                                      [-3., 0., 3.]]], device=device) / 16.0
            kernel_y = torch.tensor([[[-3., -10., -3.],
                                      [0., 0., 0.],
                                      [3., 10., 3.]]], device=device) / 16.0
            kernel_x = kernel_x.unsqueeze(0).expand(C, 1, 3, 3)
            kernel_y = kernel_y.unsqueeze(0).expand(C, 1, 3, 3)

            grad_x = F.conv2d(img, kernel_x, padding=1, groups=C)
            grad_y = F.conv2d(img, kernel_y, padding=1, groups=C)
            return torch.cat([grad_x, grad_y], dim=1)  # [B, 2C, H, W]

        elif self.operator == 'laplace':
            kernel = torch.tensor([[[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]]], device=device)
            kernel = kernel.unsqueeze(0).expand(C, 1, 3, 3)
            return F.conv2d(img, kernel, padding=1, groups=C)  # [B, C, H, W]

class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, N, N_hat):
        """
        N: 真实法向量, 形状 (B, C, H, W) 
        N_hat: 预测法向量, 形状应与 N 相同
        """
        # 创建非零 mask（按像素维度求L2范数）
        _,_,H,W = N.shape
        mask = (N.norm(p=2, dim=1, keepdim=True) > 0)  # shape: (B, 1, H, W)，True表示N非零
        mse = F.mse_loss(N, N_hat, reduction='mean') * H * W /2048 
        dot_product = torch.sum(N * N_hat, dim=1, keepdim=True)  # shape: (B, 1, H, W)
    
        # 仅在非零区域计算 loss
        loss = 1 - dot_product
        loss = loss[mask]  # 只取非零像素位置
        return loss.mean(), mse
    





def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
    mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)




class S3IM(torch.nn.Module):
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=32):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)

    def forward(self, src_vec, tar_vec):
        """
        Args:
            src_vec: [B, N, C] e.g., [batch, pixels, channels]
            tar_vec: [B, N, C]
        Returns:
            loss: scalar tensor
        """
        B, N, C = src_vec.shape
        device = src_vec.device
        patch_list_src, patch_list_tar = [], []

        for b in range(B):
            index_list = []
            for i in range(self.repeat_time):
                if i == 0:
                    tmp_index = torch.arange(N, device=device)
                else:
                    tmp_index = torch.randperm(N, device=device)
                index_list.append(tmp_index)

            res_index = torch.cat(index_list)  # [M * N]
            tar_all = tar_vec[b][res_index]    # [M*N, C]
            src_all = src_vec[b][res_index]    # [M*N, C]

            # reshape into [1, C, H, W]
            tar_patch = tar_all.permute(1, 0).reshape(1, C, self.patch_height, self.patch_width * self.repeat_time)
            src_patch = src_all.permute(1, 0).reshape(1, C, self.patch_height, self.patch_width * self.repeat_time)

            patch_list_tar.append(tar_patch)
            patch_list_src.append(src_patch)

        # Stack all batches: [B, C, H, W]
        tar_tensor = torch.cat(patch_list_tar, dim=0)
        src_tensor = torch.cat(patch_list_src, dim=0)

        # 计算 batch-wise SSIM，输出为 [B]
        ssim_scores = self.ssim_loss(src_tensor, tar_tensor)

        # 损失为 1 - mean SSIM
        loss = 1.0 - ssim_scores
        return loss



torch.manual_seed(0)

# 假设每张图片提取出 64 x 64 个像素，每个像素 3 通道
# H, W, C = 64, 32, 3
# N = H * W
# B = 4
# # 随机生成两个图像特征向量：[N, C]
# src_vec = torch.rand(B, N, C)  # 模拟重建图像
# tar_vec = torch.rand(B, N, C)  # 模拟 ground truth 图像

# # 初始化 S3IM 模块
# s3im_loss_fn = S3IM(kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=32)

# # 计算损失
# loss = s3im_loss_fn(src_vec, tar_vec)

def weighted_huber_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,          # 新增的置信度权重张量
    reduction: str = 'mean',
    delta: float = 1.0,
) -> torch.Tensor:    
    # 广播对齐所有张量
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    expanded_weight, _ = torch.broadcast_tensors(weight, input)  # 确保权重可广播
    
    # 计算逐元素误差
    diff = expanded_input - expanded_target
    abs_diff = torch.abs(diff)
    
    # Huber损失分段计算
    loss = torch.where(
        abs_diff <= delta,
        0.5 * (diff ** 2),
        delta * (abs_diff - 0.5 * delta)
    )
    
    # 应用权重
    weighted_loss = expanded_weight * loss
    
    # 汇总方式
    if reduction == 'mean':
        return torch.mean(weighted_loss)
    elif reduction == 'sum':
        return torch.sum(weighted_loss)
    elif reduction == 'none':
        return weighted_loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")