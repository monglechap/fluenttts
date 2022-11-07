import os, argparse, torch, pdb, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import hparams
from modules.model import FluentTTS
from utils.utils import *


def main(args):
    """Extract emotion embeddings of all dataset for calculating mean of emotion embeddings"""
    if not os.path.isdir(args.emb_dir):
        os.mkdir(args.emb_dir)
    mode = args.mode

    # Prepare valid or test dataset
    _, val_loader, collate_fn = prepare_dataloaders(hparams)

    # Load acoustic model
    model = FluentTTS(hparams, mode).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, betas=(0.9, 0.98), eps=1e-09)
    model, _, _, _, _ = load_checkpoint(args.checkpoint_path, model, optimizer)
    model.eval()

    # Extract
    for batch in tqdm(val_loader):
        text_padded, text_lengths, mel_padded, mel_lengths, gate_padded, \
        f0_padded, prior_padded, name, spk, emo = [x for x in batch]

        # Style embeddings
        style = model.Emo_encoder(mel_padded.cuda()).transpose(0,1).squeeze()
#        spk_emb = model.Spk_encoder(spk.cuda()).unsqueeze(0)
#        emb = torch.cat((spk_emb, style), dim=2)
#        style = model.Global_style_encoder(emb).transpose(0,1)

        # Data-specific name definition. You should change this codes for your own data structure
        # In our dataset, we divide name for 4 speakers (f1, f2, m1, m2) and 4 emotions (a, h, s, n)
        for k in range(len(style)):
            emo, spk, idx = name[k].split('_')
            if spk == 'f':
                if len(idx) == 4:
                    spk = 'f2'
                else:
                    spk = 'f1'
            elif spk == 'm':
                if len(idx) == 4:
                    spk = 'm2'
                else:
                    spk = 'm1'

            # Save
            name[k] = emo + '_' + spk + '_' + idx
            np.save(os.path.join(args.emb_dir, name[k]), style[k].detach().cpu().numpy())

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0,1')
    p.add_argument('-v', '--verbose', type=str, default='0')
    p.add_argument('-o', '--emb_dir', type=str, default='emb_dir', help='Directory for saving style embeddings')
    p.add_argument('-c', '--checkpoint_path', type=str, required=True)
    p.add_argument('-m', '--mode', type=str, help='base, prop')

    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.verbose=='0':
        import warnings
        warnings.filterwarnings("ignore")
        
    main(args)
