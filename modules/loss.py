import torch, pdb
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import get_mask_from_lengths


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        self.g = 0.2 # For guided attention loss
        
    def forward(self, pred, target, guide):
        mel_out, gate_out = pred
        mel_target, gate_target = target
        alignments, text_lengths, mel_lengths = guide
        
        mask = ~get_mask_from_lengths(mel_lengths)

        mel_target = mel_target.masked_select(mask.unsqueeze(1))
        mel_out = mel_out.masked_select(mask.unsqueeze(1))
        
        gate_target = gate_target.masked_select(mask)
        gate_out = gate_out.masked_select(mask)
            
        mel_loss = nn.L1Loss()(mel_out, mel_target)
        bce_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        guide_loss = self.guide_loss(alignments, text_lengths, mel_lengths)

        return mel_loss, bce_loss, guide_loss
    
    def guide_loss(self, alignments, text_lengths, mel_lengths):
        B, n_layers, n_heads, T, L = alignments.size()
        
        # B, T, L
        W = alignments.new_zeros(B, T, L)
        mask = alignments.new_zeros(B, T, L)
        
        for i, (t, l) in enumerate(zip(mel_lengths, text_lengths)):
            mel_seq = alignments.new_tensor( torch.arange(t).to(torch.float32).unsqueeze(-1).cuda()/t)
            text_seq = alignments.new_tensor( torch.arange(l).to(torch.float32).unsqueeze(0).cuda()/l)
            x = torch.pow(text_seq - mel_seq, 2)
            W[i, :t, :l] += alignments.new_tensor(1-torch.exp(-x/(2*(self.g**2))))

            mask[i, :t, :l] = 1
        
        # Apply guided_loss to 1 heads of the last 2 layers 
        applied_align = alignments[:, -2:, :1]
        losses = applied_align*(W.unsqueeze(1).unsqueeze(1))
        
        return torch.mean(losses.masked_select(mask.unsqueeze(1).unsqueeze(1).to(torch.bool)))


class EmotionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logit, emos):
        one_hot_name = torch.zeros((logit.size(0), logit.size(1))).cuda()

        for i in range(logit.size(0)):
            emo = emos[i].lower()
            if emo == 'h':
                one_hot_name[i][0]=1
            elif emo == 'a':
                one_hot_name[i][1]=1
            elif emo == 's':
                one_hot_name[i][2]=1
            elif emo == 'n':
                one_hot_name[i][3]=1
        loss = torch.sum(logit*one_hot_name, dim=1)
        threshold = 1e-5*torch.ones_like(loss).cuda() # For stability
        loss = torch.max(torch.sum(logit*one_hot_name, dim=1), threshold)
        loss = torch.mean(-torch.log(loss))

        return loss


