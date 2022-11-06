# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pdb
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LinearNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias: nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ConvAttention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_text_channels=512, n_att_channels=80, temperature=1.0):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
         
        self.query_proj = nn.Sequential(
            ConvNorm(n_mel_channels,
                     n_mel_channels * 2,
                     kernel_size=3,
                     bias=True,
                     w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2,
                     n_mel_channels,
                     kernel_size=1,
                     bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels,
                     n_att_channels,
                     kernel_size=1,
                     bias=True))

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels,
                     n_text_channels * 2,
                     kernel_size=3,
                     bias=True,
                     w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2,
                     n_att_channels,
                     kernel_size=1,
                     bias=True))
    
        self.key_style_proj = LinearNorm(n_text_channels, n_mel_channels)

    def forward(self, queries, keys, query_lens, mask=None, attn_prior=None,
			style_emb=None):
        """
        Args:
            queries (torch.tensor): B x C x T1 tensor
                (probably going to be mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                Final dim T2 should sum to 1
        """
        keys = keys + style_emb.transpose(1,2) 

        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2

        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        # compute log likelihood from a gaussian
        attn = -0.0005 * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None]+1e-8)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2),
                                   -float("inf"))

        attn = self.softmax(attn)  # Softmax along T2
        return attn, attn_logprob
