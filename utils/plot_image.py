import torch, random, pdb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import hparams
from text import *


def plot_melspec(target, melspec, mel_lengths, idx):
    fig, axes = plt.subplots(2, 1, figsize=(20,30))
    T = mel_lengths[idx]

    axes[0].imshow(target[idx][:,:T],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[idx][:,:T],
                   origin='lower',
                   aspect='auto')

    return fig


def plot_alignments(alignments, text, mel_lengths, text_lengths, att_type, idx):
    fig, axes = plt.subplots(hparams.n_layers, hparams.n_heads, figsize=(5*hparams.n_heads,5*hparams.n_layers))
    L, T = text_lengths[idx], mel_lengths[idx]
    n_layers, n_heads = alignments.size(1), alignments.size(2)
    
    for layer in range(n_layers):
        for head in range(n_heads):
            if att_type=='enc':
                align = alignments[idx, layer, head].contiguous()
                axes[layer,head].imshow(align[:L, :L], aspect='auto')
                axes[layer,head].xaxis.tick_top()

            elif att_type=='dec':
                align = alignments[idx, layer, head].contiguous()
                axes[layer,head].imshow(align[:T, :T], aspect='auto')
                axes[layer,head].xaxis.tick_top()

            elif att_type=='enc_dec':
                align = alignments[idx, layer, head].transpose(0,1).contiguous()
                axes[layer,head].imshow(align[:L, :T], origin='lower', aspect='auto')

    return fig

def plot_alignment(alignment): 
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()

    return fig

def plot_gate(gate_out):
    fig = plt.figure(figsize=(10,5))
    plt.plot(torch.sigmoid(gate_out))
    return fig
