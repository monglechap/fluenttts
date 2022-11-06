import os, random, pdb
from torch.utils.tensorboard import SummaryWriter

from .plot_image import *

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    writer = TTSWriter(logging_path)
            
    return writer


class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)

    def add_losses(self, total_loss, mel_loss, bce_loss, guide_loss, ctc_loss, bin_loss, emo_loss, global_step, phase, f0_loss=None):
        self.add_scalar(f'{phase}/mel_loss', mel_loss, global_step)
        self.add_scalar(f'{phase}/bce_loss', bce_loss, global_step)
        self.add_scalar(f'{phase}/guide_loss', guide_loss, global_step)
        self.add_scalar(f'{phase}/ctc_loss', ctc_loss, global_step)
        self.add_scalar(f'{phase}/bin_loss', bin_loss, global_step) 
        self.add_scalar(f'{phase}/emo_loss', emo_loss, global_step)
        self.add_scalar(f'{phase}/total_loss', total_loss, global_step)
        if f0_loss is not None:
            self.add_scalar(f'{phase}/F0_loss', f0_loss, global_step)

    def add_lr(self, current_lr, global_step, phase):
        self.add_scalar(f'{phase}/learning_rate', current_lr, global_step)

    def add_specs(self, mel_padded, mel_out, mel_lengths, global_step, phase, idx):
        mel_fig = plot_melspec(mel_padded, mel_out, mel_lengths, idx)
        self.add_figure(f'Plot/melspec', mel_fig, global_step)
        
    def add_alignments(self, enc_alignments, dec_alignments, enc_dec_alignments, 
                       text_padded, mel_lengths, text_lengths, global_step, phase, idx):
        enc_align_fig = plot_alignments(enc_alignments, text_padded, mel_lengths, text_lengths, 'enc', idx)
        self.add_figure(f'Alignment/encoder', enc_align_fig, global_step)

        dec_align_fig = plot_alignments(dec_alignments, text_padded, mel_lengths, text_lengths, 'dec', idx)
        self.add_figure(f'Alignment/decoder', dec_align_fig, global_step)

        enc_dec_align_fig = plot_alignments(enc_dec_alignments, text_padded, mel_lengths, text_lengths, 'enc_dec', idx)
        self.add_figure(f'Alignment/encoder_decoder', enc_dec_align_fig, global_step)

    def add_gates(self, gate_out, global_step, phase):
        gate_fig = plot_gate(gate_out)
        self.add_figure(f'Plot/gate_out', gate_fig, global_step)

def plot_attn(logger, soft_A, hard_A, iteration): 
    logger.add_figure("Alignment/soft_A", plot_alignment(soft_A.data.cpu().numpy()), iteration)
    logger.add_figure("Alignment/hard_A", plot_alignment(hard_A.data.cpu().numpy()), iteration)



