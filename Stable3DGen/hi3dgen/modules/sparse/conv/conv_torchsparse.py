# MIT License

# Copyright (c) Microsoft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2025] [Microsoft]
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
from .. import SparseTensor


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'torchsparse' not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
        out._spatial_cache = x._spatial_cache
        out._scale = tuple([s * stride for s, stride in zip(x._scale, self.conv.stride)])
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if 'torchsparse' not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias, transposed=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)        
        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
        out._spatial_cache = x._spatial_cache
        out._scale = tuple([s // stride for s, stride in zip(x._scale, self.conv.stride)])
        return out



