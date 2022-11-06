import torch, pdb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .init_layer import *
from .transformer import *
from utils.utils import get_mask_from_lengths, binarize_attention_parallel
from modules.saln import StyleAdaptiveLayerNorm
from .style import Emotion_encoder
from .attention import ConvAttention


class Prenet_D(nn.Module):
    '''Prenet of decoder'''
    def __init__(self, hp):
        super(Prenet_D, self).__init__()
        self.linear1 = Linear(hp.n_mel_channels, hp.dprenet_dim, w_init_gain='relu')
        self.linear2 = Linear(hp.dprenet_dim, hp.dprenet_dim, w_init_gain='relu')
        self.linear3 = Linear(hp.dprenet_dim, hp.hidden_dim)

    def forward(self, x):
        x = F.dropout(F.relu(self.linear1(x)), p=0.5, training=True)
        x = F.dropout(F.relu(self.linear2(x)), p=0.5, training=True)
        x = F.relu(self.linear3(x))
        return x


class Speaker_encoder(nn.Module):
    '''Speaker encoder based on Deep voice 3'''
    def __init__(self, hp):
        super(Speaker_encoder, self).__init__()

        self.hp = hp
        self.embedding = nn.Embedding(hp.num_spk, hp.spk_hidden_dim)
        self.linear = nn.Linear(hp.spk_hidden_dim, hp.hidden_dim)
        self.softsign = nn.Softsign()

    def forward(self, spk_id):
        embedding = self.embedding(spk_id)
        spk_emb = self.softsign(self.linear(embedding))
        return spk_emb


class Global_style_encoder(nn.Module):
    '''FC layer for combining spk & emo embeddings'''
    def __init__(self, hp):
        super(Global_style_encoder, self).__init__()
        self.hp = hp
        self.linear = nn.Linear(hp.hidden_dim*2, hp.hidden_dim)
        self.softsign = nn.Softsign()

    def forward(self, x):
        x = self.linear(x)
        x = self.softsign(x)

        return x


class F0_predictor(nn.Module):
    '''F0 predictor'''
    def __init__(self, hp):
        super(F0_predictor, self).__init__()

        self.hp = hp
        self.conv_layers = nn.ModuleList([Conv1d(hp.hidden_dim, hp.hidden_dim,
                                         kernel_size=hp.ms_kernel, padding=(hp.ms_kernel-1)//2, w_init_gain='relu') 
				                         for _ in range(hp.n_layers_lp_enc)])
        self.saln_layers = nn.ModuleList([StyleAdaptiveLayerNorm(hp.hidden_dim, hp.hidden_dim) for _ in range(hp.n_layers_lp_enc)])

        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(hp.hidden_dim, 1)

    def forward(self, x, cond, mask=None):
        x = x.transpose(1, 2) # [B, 256, L]

        for i in range(self.hp.n_layers_lp_enc):
            x = F.relu(self.conv_layers[i](x).transpose(1,2)) # [B, L, 256]
            x = self.saln_layers[i](x, cond)
            x = self.drop(x).transpose(1,2) # [B, 256, L]

        out = self.linear(x.transpose(1,2))  # [B, L, 1]
	
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(2), 0.)

        return out


class F0_encoder(nn.Module):
    '''F0 embedding'''
    def __init__(self, hp):
        super(F0_encoder, self).__init__()
        self.hp = hp
        self.conv = nn.Conv1d(1, hp.hidden_dim, kernel_size=hp.ms_kernel, padding=(hp.ms_kernel-1)//2)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x).transpose(1,2)

        return x


class Multi_style_encoder(nn.Module):
    '''Combining local F0 embeddings and global style embeddings'''
    def __init__(self, hp):
        super(Multi_style_encoder, self).__init__()
        self.hp = hp
        self.conv1 = Conv1d(hp.hidden_dim*2, hp.hidden_dim, kernel_size=hp.ms_kernel, padding=(hp.ms_kernel-1)//2, w_init_gain='relu')
        self.conv2 = Conv1d(hp.hidden_dim, hp.hidden_dim, kernel_size=hp.ms_kernel, padding=(hp.ms_kernel-1)//2, w_init_gain='relu')
        self.norm = nn.LayerNorm(hp.hidden_dim)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(hp.hidden_dim, hp.hidden_dim)
        self.softsign = nn.Softsign()

    def forward(self, x, mask=None):
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x).transpose(1,2))
        x = self.drop(self.norm(x))
        
        x = F.relu(self.conv2(x.transpose(1,2)).transpose(1,2))
        x = self.drop(self.norm(x))
        x = self.softsign(self.linear(x))

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(2), 0.)

        return x


class FluentTTS(nn.Module):
    '''FluentTTS'''
    def __init__(self, hp, mode):
        super(FluentTTS, self).__init__()
        self.hp = hp
        self.mode = mode

        # Text encoder		
        self.Layernorm = nn.LayerNorm(hp.symbols_embedding_dim)
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim) 
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)

        self.Text_encoder = nn.ModuleList([TransformerEncoderLayer(d_model=hp.hidden_dim, nhead=hp.n_heads, 
					                       ff_dim=hp.ff_dim) for _ in range(hp.n_layers)])

        # Global style encoder
        self.Spk_encoder = Speaker_encoder(hp)
        self.Emo_encoder = Emotion_encoder(hp) 
        self.Global_style_encoder = Global_style_encoder(hp)

        # Multi-style generation
        if self.mode == 'prop':
            self.Internal_aligner = ConvAttention(hp.n_mel_channels, hp.hidden_dim)	
            self.F0_predictor = F0_predictor(hp)
            self.F0_encoder = F0_encoder(hp)
            self.Multi_style_encoder = Multi_style_encoder(hp)

        # Mel decoder
        self.Prenet_D = Prenet_D(hp)
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.register_buffer('pe_d', PositionalEncoding(hp.hidden_dim*2).pe)

        self.Decoder = nn.ModuleList([TransformerDecoderLayer(d_model=hp.hidden_dim*2, nhead=hp.n_heads, 
					                  ff_dim=hp.ff_dim) for _ in range(hp.n_layers)])

        self.Projection = nn.Linear(hp.hidden_dim*2, hp.n_mel_channels)
        self.Stop = nn.Linear(hp.n_mel_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def outputs(self, text, melspec, text_lengths, mel_lengths, spk, emo, f0, prior, iteration):
        # Input data size
        B, L, T = text.size(0), text.size(1), melspec.size(2)

        # Speaker embedding (Deep voice 3)
        spk_embedding = self.Spk_encoder(spk).unsqueeze(0) # [1, B, 256]

        # Emotion embedding (Reference encoder)
        emo_embedding, emo_logit = self.Emo_encoder(melspec, logit=True)
        emo_embedding = emo_embedding.transpose(0,1)         # [1, B, 256]

        # Style embedding (FC layer)
        style_embedding = torch.cat((spk_embedding, emo_embedding), dim=2) # [1, B, 512]
        style_embedding = self.Global_style_encoder(style_embedding)       # [1, B, 256]

        # Text encoder input
        encoder_input = self.Layernorm(self.Embedding(text).transpose(0,1))
        encoder_input = encoder_input + self.alpha1*(self.pe[:L].unsqueeze(1))

        # Mel decoder input
        mel_input     = F.pad(melspec, (1,-1)).transpose(1,2)   # [B, T, 80] 
        decoder_input = self.Prenet_D(mel_input).transpose(0,1) # [T, B, 256]
       
        # Masks
        text_mask = get_mask_from_lengths(text_lengths) 
        mel_mask = get_mask_from_lengths(mel_lengths)
        diag_mask = torch.triu(melspec.new_ones(T,T)).transpose(0, 1) 
        diag_mask[diag_mask == 0] = -float('inf')
        diag_mask[diag_mask == 1] = 0
    
        # Text encoder
        memory = encoder_input
        enc_alignments = []
        for layer in self.Text_encoder:
            memory, enc_align = layer(memory, src_key_padding_mask=text_mask) # [L,B,256]
            enc_alignments.append(enc_align.unsqueeze(1))
        enc_alignments = torch.cat(enc_alignments, 1)

        # Internal aligner
        if self.mode == 'prop':
            soft_A, attn_logprob = self.Internal_aligner(mel_input.transpose(1,2), memory.permute(1,2,0), mel_lengths, 
				                                         text_mask.unsqueeze(-1), prior.transpose(1,2), 
				                                         style_embedding.detach().transpose(0,1)) # [B, 1, T, L] [B, L, T]
            hard_A = binarize_attention_parallel(soft_A, text_lengths, mel_lengths)
        else:
            soft_A, hard_A, attn_logprob = torch.zeros(B, 1, T, L).cuda(), torch.zeros(B, 1, T, L).cuda(), torch.zeros(B, L, T).cuda()
    
        # Multi-style generation
        if iteration > self.hp.local_style_step and self.mode == 'prop':
            # Phoneme-level target F0
            aligned_f0 = torch.bmm(hard_A.squeeze().transpose(1,2), f0.unsqueeze(2))         # [B, L, 1]
            nonzero    = torch.count_nonzero(hard_A.squeeze().transpose(1,2), dim=2)         # [B, L]
            aligned_f0 = torch.div(aligned_f0.squeeze(2), nonzero).nan_to_num().unsqueeze(2) # [B, L, 1]

            # Phoneme-level predicted F0
            pred_f0 = self.F0_predictor(memory.detach().transpose(0,1), 
                                        style_embedding.detach().transpose(0,1), text_mask)  # [B, L, 1]

            f0_emb = self.F0_encoder(aligned_f0)

            expand_style_enc = style_embedding.expand(memory.size(0), -1, -1).transpose(0,1) # [B, L, 256]
            local_style_emb = torch.cat((expand_style_enc, f0_emb), dim=2)                   # [B, L, 512]
            local_style_emb = self.Multi_style_encoder(local_style_emb).transpose(0,1)       # [L, B, 256]
        
            memory = torch.cat((memory, local_style_emb), dim=2)

        # For initial training part and Baseline
        else: 
            expand_style_enc = style_embedding.expand(memory.size(0), -1, -1) # [L, B, 256]
            memory = torch.cat((memory, expand_style_enc), dim=2)             # [L, B, 512]

        # Mel decoder
        expand_style_dec = style_embedding.expand(decoder_input.size(0), -1, -1)                             # [T, B, 256]
        tgt = torch.cat((decoder_input, expand_style_dec), dim=2) + self.alpha2*(self.pe_d[:T].unsqueeze(1)) # [T, B, 512]

        dec_alignments, enc_dec_alignments = [], []
        for layer in self.Decoder:
            tgt, dec_align, enc_dec_align = layer(tgt,
                                                  memory,
                                                  tgt_mask=diag_mask,
                                                  tgt_key_padding_mask=mel_mask,
                                                  memory_key_padding_mask=text_mask)
            dec_alignments.append(dec_align.unsqueeze(1))
            enc_dec_alignments.append(enc_dec_align.unsqueeze(1))
        dec_alignments = torch.cat(dec_alignments, 1)
        enc_dec_alignments = torch.cat(enc_dec_alignments, 1)

        # Projection + Stop token
        mel_out = self.Projection(tgt.transpose(0, 1)).transpose(1, 2)
        gate_out = self.Stop(mel_out.transpose(1, 2)).squeeze(-1)
     
        # Return
        if iteration > self.hp.local_style_step and self.mode == 'prop':
            return mel_out, enc_alignments, dec_alignments, enc_dec_alignments, gate_out, \
                   soft_A, hard_A, attn_logprob, emo_logit, aligned_f0, pred_f0
        else:
            return mel_out, enc_alignments, dec_alignments, enc_dec_alignments, gate_out, \
                   soft_A, hard_A, attn_logprob, emo_logit
 
   
    def forward(self, text, melspec, gate, f0, prior, text_lengths, mel_lengths, 
                criterion, criterion_emo, criterion_ctc, criterion_bin, criterion_f0,
                spk, emo, iteration, valid=None):
        # Input data
        text    = text[:,:text_lengths.max().item()]
        melspec = melspec[:,:,:mel_lengths.max().item()]
        gate    = gate[:, :mel_lengths.max().item()]
        f0      = f0[:, :mel_lengths.max().item()]
        prior   = prior[:, :text_lengths.max().item(), :mel_lengths.max().item()]
        
        # Model outputs
        outputs = self.outputs(text, melspec, text_lengths, mel_lengths, spk, emo, f0, prior, iteration)
        
        # Parse
        mel_out = outputs[0]
        enc_dec_alignments = outputs[3]
        gate_out = outputs[4]
        soft_A, hard_A, attn_logprob = outputs[5], outputs[6], outputs[7]      
        emo_logit = outputs[8]
        
        # TTS loss
        mel_loss, bce_loss, guide_loss = criterion((mel_out, gate_out),
                                                   (melspec, gate),
                                                   (enc_dec_alignments, text_lengths, mel_lengths))
        
        # Internal aligner loss
        if self.mode == 'prop':
            ctc_loss = criterion_ctc(attn_logprob, text_lengths, mel_lengths)
        
            if iteration < self.hp.bin_loss_enable_steps:
                bin_loss_weight = 0.
            else:
                bin_loss_weight = self.hp.kl_scale
            bin_loss = criterion_bin(hard_A, soft_A) * bin_loss_weight
        else:
            ctc_loss, bin_loss = torch.FloatTensor([0]).cuda(), torch.FloatTensor([0]).cuda()

        # Emotion classification loss
        emo_loss = criterion_emo(emo_logit.squeeze(1), emo)
        emo_loss = emo_loss * self.hp.emo_scale

        # F0 loss
        if iteration > self.hp.local_style_step and self.mode == 'prop':
            aligned_f0, pred_f0 = outputs[9], outputs[10]
            f0_loss = criterion_f0(pred_f0, aligned_f0)
            f0_loss = f0_loss * self.hp.f0_scale
            if valid:
                return mel_loss, bce_loss, guide_loss, ctc_loss, bin_loss, f0_loss, emo_loss, outputs, mel_lengths
            else:
                return mel_loss, bce_loss, guide_loss, ctc_loss, bin_loss, f0_loss, emo_loss
        else:
            if valid:
                return mel_loss, bce_loss, guide_loss, ctc_loss, bin_loss, emo_loss, outputs, mel_lengths
            else:
                return mel_loss, bce_loss, guide_loss, ctc_loss, bin_loss, emo_loss


    def inference(self, text, emo_embedding, spk_id, f0_mean, f0_std, 
                  max_len=1024, mode=None, slide=False, start=None, end=None, hz=None):
        # Input data size
        (B, L), T = text.size(), max_len
        
        # Speaker embedding
        spk_embedding = self.Spk_encoder(spk_id).unsqueeze(0) # [1, 1, 256]

        # Emotion embedding (from reference mel or mean style embedding)
        emo_embedding = emo_embedding

        # Style embedding
        style_embedding = torch.cat((spk_embedding, emo_embedding), dim=2) # [1, 1, 512]
        style_embedding = self.Global_style_encoder(style_embedding)       # [1, 1, 256]

        # Text encoder input
        encoder_input = self.Layernorm(self.Embedding(text).transpose(0,1))
        encoder_input = encoder_input + self.alpha1*(self.pe[:L].unsqueeze(1))

        # Masks
        text_mask  = text.new_zeros(1, L).to(torch.bool)
        mel_mask = text.new_zeros(1, T).to(torch.bool)
        diag_mask = torch.triu(text.new_ones(T, T)).transpose(0, 1).contiguous()
        diag_mask[diag_mask == 0] = -1e9
        diag_mask[diag_mask == 1] = 0
        diag_mask = diag_mask.float()
        
        # Text encoder
        memory = encoder_input
        enc_alignments = []
        for layer in self.Text_encoder:
            memory, enc_align = layer(memory, src_key_padding_mask=text_mask) # [L, 1, 256]
            enc_alignments.append(enc_align)
        enc_alignments = torch.cat(enc_alignments, dim=0)
        
        if mode == 'prop':
            # Multi-style generation
            pred_f0 = self.F0_predictor(memory.transpose(0,1), style_embedding.transpose(0,1), text_mask) # [1, L, 1]
       
            # Dynamic-level F0 control
            # Case 1. Word or phoneme-level
            if start is not None:
                print(f'Word/Phoneme-level F0 control | hz = {hz}')
                norm_hz = hz / f0_std
                pred_f0[:, start:end] = pred_f0[:, start:end] + norm_hz
            # Case 2. Utterance-level
            elif hz is not None:
                print(f'Utterance-level, F0 shift | hz = {hz}')
                norm_hz = hz / f0_std
                pred_f0 = pred_f0 + norm_hz

            f0_emb = self.F0_encoder(pred_f0) # [1, L, 256]

            expand_style_enc = style_embedding.expand(memory.size(0), -1, -1).transpose(0,1) # [1, L, 256]
            local_style_emb = torch.cat((expand_style_enc, f0_emb), dim=2)                   # [1, L, 512]
            local_style_emb = self.Multi_style_encoder(local_style_emb).transpose(0,1)       # [L, 1, 256]

            memory = torch.cat((memory, local_style_emb), dim=2) # [L, 1, 512]
        
        else:
            expand_style_enc = style_embedding.expand(memory.size(0), -1, -1) # [1, L, 256]
            memory = torch.cat((memory, expand_style_enc), dim=2) # [L, 1, 512]        

        # Decoder inputs
        mel_input = text.new_zeros(1,
                                   self.hp.n_mel_channels,
                                   max_len).to(torch.float32)
        dec_alignments = text.new_zeros(self.hp.n_layers,
                                        self.hp.n_heads,
                                        max_len,
                                        max_len).to(torch.float32)
        enc_dec_alignments = text.new_zeros(self.hp.n_layers,
                                            self.hp.n_heads,
                                            max_len,
                                            text.size(1)).to(torch.float32)

        # Autoregressive generation
        stop = [] # Stop token

        if not slide:
            print('AR generation')
            for i in range(max_len):
                # Preparation
                decoder_input = self.Prenet_D(mel_input.transpose(1,2).contiguous()).transpose(0,1).contiguous()
                expand_style_dec = style_embedding.expand(decoder_input.size(0), -1, -1)
                tgt = torch.cat((decoder_input, expand_style_dec), dim=2) + self.alpha2*(self.pe_d[:T].unsqueeze(1))

                # Decoder
                for j, layer in enumerate(self.Decoder):
                    tgt, dec_align, enc_dec_align = layer(tgt,
                                                        memory,
                                                        tgt_mask=diag_mask,
                                                        tgt_key_padding_mask=mel_mask,
                                                        memory_key_padding_mask=text_mask)
                
                    dec_alignments[j, :, i] = dec_align[0, :, i]
                    enc_dec_alignments[j, :, i] = enc_dec_align[0, :, i]
                
                # Outputs
                mel_out = self.Projection(tgt.transpose(0,1).contiguous())
                stop.append(torch.sigmoid(self.Stop(mel_out[:,i]))[0,0].item()) 
                
                # Store generated frame
                if i < max_len - 1:
                    mel_input[0, :, i+1] = mel_out[0, i] # [1,80,1024]

                # Break point
                if stop[-1]>0.5:
                    break
        else:
            print('AR generation with sliding window attention')
            # Sliding window attention initialization
            left_idx   = 0
            right_idx  = self.hp.sliding_window[1] + 1 # 5
            win_center = 0
            update_cnt = 0 # update center when reaching 3
            
            # Each mel spectrogram frame
            for i in range(max_len):
                # Slice encoder output by sliding window size
#                print(i, left_idx, right_idx, win_center)
                current_memory = memory[left_idx:right_idx + 1, :, :] # [6, 1, 256]
                slide_text_mask = text.new_zeros(1, right_idx-left_idx+1).to(torch.bool) # [1, 6]

                # Preparation
                decoder_input = self.Prenet_D(mel_input.transpose(1,2).contiguous()).transpose(0,1).contiguous()
                expand_style_dec = style_embedding.expand(decoder_input.size(0), -1, -1)
                tgt = torch.cat((decoder_input, expand_style_dec), dim=2) + self.alpha2*(self.pe_d[:T].unsqueeze(1))

                # Decoder
                for j, layer in enumerate(self.Decoder):
                    tgt, dec_align, enc_dec_align = layer(tgt,
                                                        current_memory,
                                                        tgt_mask=diag_mask,
                                                        tgt_key_padding_mask=mel_mask,
                                                        memory_key_padding_mask=slide_text_mask)
                
                    dec_alignments[j, :, i] = dec_align[0, :, i]
                    enc_dec_alignments[j, :, i, left_idx:right_idx + 1] = enc_dec_align[0, :, i]
                
                # Outputs
                mel_out = self.Projection(tgt.transpose(0,1).contiguous())
                stop.append(torch.sigmoid(self.Stop(mel_out[:,i]))[0,0].item()) 
                
                # Store generated frame
                if i < max_len - 1:
                    mel_input[0, :, i+1] = mel_out[0, i] # [1,80,1024]

                # Break point
                if stop[-1]>0.5:
                    break

                # Sliding window update
                # Merge enc_dec_alignment of i-th frame for all layer and head
                tmp_att = torch.sum(enc_dec_alignments, dim=(0,1))[i, left_idx:right_idx+1]

                # Get attention centroid
                range_tensor = torch.arange(left_idx, right_idx+1).cuda()
                att_centroid = torch.floor(torch.sum(tmp_att * range_tensor) / (self.hp.n_layers*self.hp.n_heads)).int()

                # Update window center 
                if att_centroid > win_center:
                    update_cnt += 1
                    if update_cnt == 3:
                        update_cnt = 0
                        win_center += 1
                        left_idx   = max(0, win_center + self.hp.sliding_window[0])
                        right_idx  = min(win_center + self.hp.sliding_window[1], L-2) 
             
        return mel_out.transpose(1,2), enc_alignments, dec_alignments, enc_dec_alignments, stop


