import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pdb
import random
from utils.utils import *


class Emotion_encoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.encoder = ReferenceEncoder(hparams)

    def forward(self, inputs, logit=None):
        emo_embed, emo_logit = self.encoder(inputs)
        
        if logit: 
            return emo_embed, emo_logit
        else: 
            return emo_embed


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''
    def __init__(self,hparams):
        super().__init__()
        self.ref_enc_filters = hparams.ref_enc_filters
        self.n_mel_channels = hparams.n_mel_channels
        self.E = hparams.E
        K = len(self.ref_enc_filters)
        filters = [1] + self.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=self.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(self.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=self.ref_enc_filters[-1] * out_channels,
                          hidden_size=self.E // 2,
                          batch_first=True)
        self.out_fc = nn.Linear(self.E//2, self.E)
        self.softsign = nn.Softsign()
        self.emo_logit_extractor = torch.nn.Sequential(torch.nn.Linear(self.E, self.E//2),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(self.E//2, hparams.num_emo),
                                             torch.nn.Softmax(dim=-1))

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]
        out = out.squeeze(0)
        out = self.softsign(self.out_fc(out.unsqueeze(1)))
        emo_logit = self.emo_logit_extractor(out)
        return out, emo_logit

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


