import os, sys, argparse, librosa, torch, pdb, glob, warnings
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from g2pK.g2pk.g2pk import G2p

import hparams
from text import *
from text.symbols import symbols
from text.cleaners import basic_cleaners
from modules.model import FluentTTS
from utils.utils import *
from utils.data_utils import process_meta, create_id_table
from model_hifigan import Generator
from layers import TacotronSTFT


# Prepare text preprocessing
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}
stft = TacotronSTFT()

def text2seq(text):
    """Text preprocessing"""
    sequence=[symbol_to_id['^']]
    sequence.extend(text_to_sequence(text, hparams.text_cleaners))
    sequence.append(symbol_to_id['~'])
    return sequence

def text2seq_target(text):
    """For word & phoneme-level f0 control"""
    sequence=[]
    sequence.extend(text_to_sequence(text, hparams.text_cleaners))
    return sequence

def get_mel(filename):
    """Prepare mel spectrogram from reference wav"""
    wav, sr = librosa.load(filename, sr=hparams.sampling_rate)
    wav = librosa.effects.trim(wav, top_db=23, frame_length=1024, hop_length=256)[0]
    wav = torch.FloatTensor(wav.astype(np.float32))
    melspec, _ = stft.mel_spectrogram(wav.unsqueeze(0))
    return melspec.squeeze(0)


def synthesize(args, style_list):
    # Load Acoustic model
    mode = args.mode

    model = FluentTTS(hparams, mode).cuda()
    model, _, _, _, _ = load_checkpoint(args.checkpoint_path, model, None)
    model.cuda().eval()

    # Load Vocoder
    generator = Generator(hparams).cuda()
    state_dict_g = torch.load(args.vocoder)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    # Prepare speaker ID and input text
    _, speakers, _ = process_meta(hparams.validation_files)
    sid_dict = create_id_table(speakers) # {'f1': 0, 'f2': 1, 'm1': 2, 'm2': 3}
    
    g2p = G2p()
    with open(hparams.inference_files, 'r') as f:
        text_list = f.readlines()
    idx = 0 # Text number
   
    # Inference for each utterance
    for text in text_list: 
        # Word & Phoneme-level F0 control
        if args.control == 'pho':
            text, target = text.strip('\n').split('|')
            text, target = g2p(text), g2p(target)
            print(f'{text}|{target}')

            src_seq = np.array(text2seq(text))
            tgt_seq = np.array(text2seq_target(target))
            # Find index of target sequence in source sequence
            i = 0
            while True:
                if np.equal(src_seq[i:len(tgt_seq)+i], tgt_seq).all():
                    start, end = i, len(tgt_seq) + i
                    break
                else:
                    i += 1
            print(start, end)
        # Utterance-level
        else:
            text = text.strip('\n')
            text = g2p(text)
            print(text)

        # Inference for each style vectors
        for style_path in style_list:
            # Text sequence
            sequence = np.array(text2seq(text))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            
            # Style vector of mean or i2i or reference wav
            if style_path[-4:] == '.npy':
                style = torch.from_numpy(np.load(style_path)).view(1,1,-1).cuda()
                spk = style_path.split('_')[-1][:-4]
                emo = style_path.split('_')[-2][0]
            elif style_path[-4:] == '.wav':
                ref_mel, _, _ = get_mel(style_path)
                ref_mel = torch.from_numpy(ref_mel).unsqueeze(0).float().cuda()
                style = model.Emo_encoder(ref_mel, logit=False).transpose(0,1)
                spk, emo = args.spk, args.emo

            # Load mean and std of f0
            ms_path = os.path.join(hparams.data_path, 'mean_std/')
            mean_name = 'mean_' + spk + '_' + emo + '.npy'
            std_name  = 'std_' + spk + '_' + emo + '.npy'
            f0_mean = torch.from_numpy(np.load(os.path.join(ms_path, mean_name))).cuda()
            f0_std = torch.from_numpy(np.load(os.path.join(ms_path, std_name))).cuda()

            # Speaker ID
            spk_id = sid_dict[spk]
            spk_id = torch.LongTensor([spk_id]).cuda()

            # Inference
            with torch.no_grad():
                # Word & Phoneme-level F0 control
                if args.control == 'pho':
                    melspec, enc_alignments, dec_alignments, enc_dec_alignments, stop = model.inference(sequence, style, spk_id, f0_mean, f0_std, max_len=1024, mode=mode, slide=args.slide, start=start, end=end, hz=args.hz)
                # Uttr or not controlling F0
                else:
                    melspec, enc_alignments, dec_alignments, enc_dec_alignments, stop = model.inference(sequence, style, spk_id, f0_mean, f0_std, max_len=1024, mode=mode, slide=args.slide, hz=args.hz)

                T=len(stop)
                melspec = melspec[:,:,:T]   
                    
                # Waveform generation
                y_g_hat = generator(melspec)
                audio = y_g_hat.squeeze()
                audio = audio*32768
                audio = audio.detach().cpu().numpy().astype('int16')           
                name = style_path.split('/')[-1][:-4][4:]
                output = args.out_dir + '/' + str(idx) + '_' + args.mode + '_' + name + '.wav'
                write(output, hparams.sampling_rate, audio)
                print(output) 

                # Plot mel spectrogram
                plot_mel = melspec.squeeze(0).detach().cpu().numpy()
                plt.figure(figsize=(8,6))
                plt.imshow(plot_mel, origin='lower', aspect='auto')
                name = output[:-4] + '.png'
                plt.savefig(name)
                plt.close()
                
        idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('-c', '--checkpoint_path', type=str)
    parser.add_argument('-v', '--vocoder', type=str, help='ckpt path of Hifi-GAN')
    parser.add_argument('-s', '--style_dir', type=str, default='emb_mean_dir')
    parser.add_argument('-o', '--out_dir', type=str, default='generated_files')
    parser.add_argument('-m', '--mode', type=str, help='base, prop')
    parser.add_argument('--control', type=str, default=None, help='uttr, pho')
    parser.add_argument('--hz', type=float, default=None, help='value to modify f0')
    parser.add_argument('--ref_dir', type=str, default=None, help='use when using referece wav')
    parser.add_argument('--spk', type=str, default='f2', help='use when using reference wav')
    parser.add_argument('--emo', type=str, default='a', help='use when using reference wav') 
    parser.add_argument('--slide', action = 'store_true')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.ref_dir is not None:
        style_path = args.ref_dir + '**/*.wav'
    else:
        style_path = args.style_dir + '**/*.npy'

    style_list = [file for file in glob.glob(style_path, recursive=True)]
    print(f'Number of style vector: {len(style_list)}')

    warnings.filterwarnings('ignore')

    synthesize(args, style_list)

