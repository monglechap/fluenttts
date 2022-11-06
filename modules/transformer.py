import torch, pdb
import torch.nn as nn
import torch.nn.functional as F

from .init_layer import *


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model, nhead, ff_dim,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ffn = nn.Sequential(
				Linear(d_model, ff_dim, w_init_gain='relu'),
				nn.ReLU(),
				nn.Dropout(dropout),
				Linear(ff_dim, d_model)
				)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention
        slf_attn_out, enc_align = self.self_attn(src,
                                             src,
                                             src,
                                             attn_mask=src_mask,
                                             key_padding_mask=src_key_padding_mask)
        # Add & Norm
        src = src + self.dropout(slf_attn_out)
        src = self.norm1(src)

        # FFN
        ffn_out = self.ffn(src)

        # Add & Norm
        src = src + self.dropout(ffn_out)
        src = self.norm2(src)

        return src, enc_align


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model, nhead, ff_dim,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cros_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ffn = nn.Sequential(
				Linear(d_model, ff_dim, w_init_gain='relu'),
				nn.ReLU(),
				nn.Dropout(dropout),
				Linear(ff_dim, d_model)
				)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self attention
        slf_attn_out, dec_align = self.self_attn(tgt,
                                                 tgt,
                                                 tgt,
                                                 attn_mask=tgt_mask,
                                                 key_padding_mask=tgt_key_padding_mask)
        # Add & Norm
        tgt = tgt + self.dropout(slf_attn_out)
        tgt = self.norm1(tgt)
        
        # Cross attention
        crs_attn_out, enc_dec_align = self.cros_attn(tgt,
                                                     memory,
                                                     memory,
                                                     attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)
        # Add & Norm
        tgt = tgt + self.dropout(crs_attn_out)
        tgt = self.norm2(tgt)
        
        # FFN
        ffn_out = self.ffn(tgt)

        # Add & Norm
        tgt = tgt + self.dropout(ffn_out)
        tgt = self.norm3(tgt)
        
        return tgt, dec_align, enc_dec_align


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self._get_pe_matrix(d_model, max_len))

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)
    
    def _get_pe_matrix(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        
        return pe
        
