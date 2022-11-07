import os, argparse, pdb, random
import torch
import torch.nn as nn
import torch.nn.functional as F

import hparams
from modules.model import FluentTTS
from modules.loss import TransformerLoss, EmotionLoss
from modules.attn_loss_function import AttentionCTCLoss, AttentionBinarizationLoss
from text import *
from utils.utils import *
from utils.writer import get_writer, plot_attn


def validate(model, criterion, criterion_emo, criterion_ctc, criterion_bin, criterion_f0, val_loader, writer, iteration):
    model.eval()
    with torch.no_grad():
        n_data, val_loss = 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])

            # Get mini-batch
            text_padded, text_lengths, mel_padded, mel_lengths, gate_padded, \
            f0_padded, prior_padded, name, spk, emo = [reorder_batch(x, 1) for x in batch]
           
            # Forward (w/ multi-style generation)
            if iteration > hparams.local_style_step and model.mode == 'prop':			
                mel_loss, bce_loss, guide_loss, \
                ctc_loss, bin_loss, f0_loss, emo_loss, outputs, mel_lengths = model(text_padded, mel_padded, gate_padded, f0_padded, prior_padded, text_lengths, mel_lengths,
                                                                       criterion, criterion_emo, criterion_ctc, criterion_bin, criterion_f0,
                                                                       spk, emo, iteration, valid=True)
            
                mel_loss, bce_loss, guide_loss, \
				ctc_loss, bin_loss, f0_loss, emo_loss = [torch.mean(x) for x in [mel_loss, bce_loss, guide_loss, \
				                                                                 ctc_loss, bin_loss, f0_loss, emo_loss]]
                sub_loss = mel_loss + bce_loss + guide_loss + ctc_loss + bin_loss + f0_loss + emo_loss

            # Forward (w/o multi-style generation)
            else:
                mel_loss, bce_loss, guide_loss, \
                ctc_loss, bin_loss, emo_loss, outputs, mel_lengths = model(text_padded, mel_padded, gate_padded, f0_padded, prior_padded, text_lengths, mel_lengths,
                                                              criterion, criterion_emo, criterion_ctc, criterion_bin, criterion_f0,
                                                              spk, emo, iteration, valid=True)
            
                mel_loss, bce_loss, guide_loss, \
				ctc_loss, bin_loss, emo_loss = [torch.mean(x) for x in [mel_loss, bce_loss, guide_loss, \
				                                                        ctc_loss, bin_loss, emo_loss]]
                sub_loss = mel_loss + bce_loss + guide_loss + ctc_loss + bin_loss + emo_loss

            val_loss += sub_loss.item() * len(batch[0])

        val_loss /= n_data

    # Tensorboard
    if iteration > hparams.local_style_step and model.mode == 'prop':
        writer.add_losses(val_loss, mel_loss.item(), bce_loss.item(), guide_loss.item(),
                          ctc_loss.item(), bin_loss.item(), emo_loss.item(),
                          iteration, 'Validation', f0_loss.item())
    else:
        writer.add_losses(val_loss, mel_loss.item(), bce_loss.item(), guide_loss.item(),
                          ctc_loss.item(), bin_loss.item(), emo_loss.item(),
                          iteration, 'Validation')

    # Plot
    mel_out, enc_alignments, dec_alignments, enc_dec_alignments, gate_out = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
    idx = random.randint(0, len(mel_out)-1)

    writer.add_specs(mel_padded.detach().cpu(),
                     mel_out.detach().cpu(),
                     mel_lengths.detach().cpu(),
                     iteration, 'Validation', idx)
 
    writer.add_alignments(enc_alignments.detach().cpu(),
                          dec_alignments.detach().cpu(),
                          enc_dec_alignments.detach().cpu(),
                          text_padded.detach().cpu(),
                          mel_lengths.detach().cpu(),
                          text_lengths.detach().cpu(),
                          iteration, 'Validation', idx)

    if model.mode == 'prop':
        soft_A, hard_A = outputs[5], outputs[6]
        soft_A = soft_A[idx].squeeze().transpose(0,1)[:text_lengths[idx], :mel_lengths[idx]]
        hard_A = hard_A[idx].squeeze().transpose(0,1)[:text_lengths[idx], :mel_lengths[idx]]

        plot_attn(writer, soft_A, hard_A, iteration)

    writer.add_gates(gate_out[idx].detach().cpu(), iteration, 'Validation')

    print(f'\nValidation: {iteration} | loss {val_loss:.4f}')

    model.train()
    
     
def main(args):
    # Preparation
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    mode = args.mode
    
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams)
    model = FluentTTS(hparams, mode).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-6)
    model, optimizer, last_epoch, learning_rate, iteration = load_checkpoint(args.checkpoint_path, model, optimizer)

    criterion       = TransformerLoss()
    criterion_emo   = EmotionLoss()
    criterion_f0    = nn.L1Loss()
    criterion_ctc   = AttentionCTCLoss()
    criterion_bin   = AttentionBinarizationLoss()

    writer = get_writer(args.outdir, 'logdir')

    loss = 0
    num_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param += param.numel()
    print(f'Model parameters: {num_param/1000000:.2f}M')

    # Training
    model.train()
    print("Training start!")
    for epoch in range(max(0, last_epoch), hparams.training_epochs):
        for i, batch in enumerate(train_loader):
            # Get mini-batch
            text_padded, text_lengths, mel_padded, mel_lengths, gate_padded, \
            f0_padded, prior_padded, name, spk, emo = [reorder_batch(x, hparams.n_gpus) for x in batch]

            # Forward (w/ multi-style generation)
            if iteration > hparams.local_style_step and mode == 'prop':			
                mel_loss, bce_loss, guide_loss, \
                ctc_loss, bin_loss, f0_loss, emo_loss = model(text_padded, mel_padded, gate_padded, f0_padded, prior_padded, text_lengths, mel_lengths,
                                                              criterion, criterion_emo, criterion_ctc, criterion_bin, criterion_f0,
                                                              spk, emo, iteration)
            
                mel_loss, bce_loss, guide_loss, \
				ctc_loss, bin_loss, f0_loss, emo_loss = [torch.mean(x) for x in [mel_loss, bce_loss, guide_loss, \
				                                                                 ctc_loss, bin_loss, f0_loss, emo_loss]]
                sub_loss = mel_loss + bce_loss + guide_loss + ctc_loss + bin_loss + f0_loss + emo_loss

            # Forward (w/o multi-style generation)
            else:
                mel_loss, bce_loss, guide_loss, \
                ctc_loss, bin_loss, emo_loss = model(text_padded, mel_padded, gate_padded, f0_padded, prior_padded, text_lengths, mel_lengths,
                                                     criterion, criterion_emo, criterion_ctc, criterion_bin, criterion_f0,
                                                     spk, emo, iteration)
            
                mel_loss, bce_loss, guide_loss, \
				ctc_loss, bin_loss, emo_loss = [torch.mean(x) for x in [mel_loss, bce_loss, guide_loss, \
				                                                        ctc_loss, bin_loss, emo_loss]]
                sub_loss = mel_loss + bce_loss + guide_loss + ctc_loss + bin_loss + emo_loss

            # Backward
            sub_loss.backward()
            loss = loss+sub_loss.item()
            
            iteration += 1
            lr_scheduling(optimizer, iteration)
            nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            print(f"[Epoch {epoch}] Train: {iteration} step | loss {loss:.4f}", end='\r')

            # Tensorboard
            if iteration > hparams.local_style_step + 1 and mode == 'prop':
                writer.add_losses(loss, mel_loss.item(), bce_loss.item(), guide_loss.item(), 
                                  ctc_loss.item(), bin_loss.item(), emo_loss.item(),
                                  iteration, 'Train', f0_loss.item())
            else:
                writer.add_losses(loss, mel_loss.item(), bce_loss.item(), guide_loss.item(), 
                                  ctc_loss.item(), bin_loss.item(), emo_loss.item(),
                                  iteration, 'Train')

            loss = 0

            # Validation & Save
            if iteration % hparams.iters_per_validation == 0:
                validate(model, criterion, criterion_emo, criterion_ctc, criterion_bin, criterion_f0,
                         val_loader, writer, iteration)

            if iteration % hparams.iters_per_checkpoint == 0:
                save_checkpoint(model, optimizer, epoch, hparams.lr, iteration, filepath=f'{args.outdir}/logdir')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0,1')
    p.add_argument('-v', '--verbose', type=str, default='0')
    p.add_argument('-c', '--checkpoint_path', default=None)
    p.add_argument('-o', '--outdir', default='outdir')
    p.add_argument('-m', '--mode', type=str, help='base, prop')
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.verbose=='0':
        import warnings
        warnings.filterwarnings("ignore")
        
    main(args)
