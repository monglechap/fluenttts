import os, random, torch, pdb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from numba import jit, prange
import numpy as np

import hparams
from .data_utils import TextMelSet, TextMelCollate
from text import *


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def prepare_dataloaders(hparams):
    trainset = TextMelSet(hparams.training_files, hparams)
    valset = TextMelSet(hparams.validation_files, hparams)
    collate_fn = TextMelCollate()

    train_loader = DataLoader(trainset,
                              num_workers=hparams.n_gpus-1,
                              shuffle=True,
                              batch_size=hparams.batch_size, 
                              drop_last=True, 
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(valset,
                            batch_size=hparams.batch_size//hparams.n_gpus,
                            collate_fn=collate_fn)
   
    spk_label = [key for key in valset.sid_dict]
    emo_label = [key for key in valset.eid_dict]
    print(spk_label)
    print(emo_label)

    return train_loader, val_loader, collate_fn


def load_checkpoint(checkpoint_path, model, optimizer):
    if checkpoint_path == None: return model, optimizer, 0, hparams.lr, 0
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu') # pretrained
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,v in checkpoint_dict['state_dict'].items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    learning_rate = checkpoint_dict['learning_rate']
    epoch = checkpoint_dict['epoch']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))

    if optimizer is None: return model, optimizer, epoch, learning_rate, iteration

    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return model, optimizer, epoch, learning_rate, iteration


def save_checkpoint(model, optimizer, epoch, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')

    
def lr_scheduling(opt, step, init_lr=hparams.lr, warmup_steps=hparams.warmup_steps):
    opt.param_groups[0]['lr'] = init_lr * min(step ** -0.5, (warmup_steps ** -1.5) * step)
    return


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len))
    mask = (lengths.unsqueeze(1) <= ids.cuda()).to(torch.bool)
    return mask


def get_mask(lengths):
    mask = torch.zeros(len(lengths), torch.max(lengths)).cuda()
    for i in range(len(mask)):
        mask[i] = torch.nn.functional.pad(torch.arange(1,lengths[i]+1),[0,torch.max(lengths)-lengths[i]],'constant')
    return mask.cuda()


def reorder_batch(x, n_gpus):
    assert (len(x)%n_gpus)==0, 'Batch size must be a multiple of the number of GPUs.'
    if isinstance(x, list):
        return x
    new_x = x.new_zeros(x.size())
    chunk_size = x.size(0)//n_gpus
    
    for i in range(n_gpus):
        new_x[i::n_gpus] = x[i*chunk_size:(i+1)*chunk_size]
    
    return new_x.cuda()


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out

def binarize_attention_parallel(attn, in_lens, out_lens):
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
    return torch.from_numpy(attn_out).cuda()


