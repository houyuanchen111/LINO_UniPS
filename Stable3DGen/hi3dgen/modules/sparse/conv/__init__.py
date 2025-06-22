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
from .. import BACKEND


SPCONV_ALGO = 'auto'    # 'auto', 'implicit_gemm', 'native'

def __from_env():
    import os
        
    global SPCONV_ALGO
    env_spconv_algo = os.environ.get('SPCONV_ALGO')
    if env_spconv_algo is not None and env_spconv_algo in ['auto', 'implicit_gemm', 'native']:
        SPCONV_ALGO = env_spconv_algo
    print(f"[SPARSE][CONV] spconv algo: {SPCONV_ALGO}")
        

__from_env()

if BACKEND == 'torchsparse':
    from .conv_torchsparse import *
elif BACKEND == 'spconv':
    from .conv_spconv import *
