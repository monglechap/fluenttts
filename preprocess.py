import os, librosa, torch, pdb, sys
import numpy as np
import pyworld as pw
from statistics import mean
from tqdm import tqdm
from scipy.stats import betabinom

import hparams
from text import *
from text.cleaners import basic_cleaners
from text.symbols import symbols
from layers import TacotronSTFT
from utils.data_utils import process_meta, create_id_table

stft = TacotronSTFT()

### Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}

### Prepare data path
path = 'Data path to save preprocessed files'
for fname in ('mels', 'texts', 'f0', 'mean_std', 'alignment_priors'):
    os.makedirs(os.path.join(path, fname), exist_ok=True)

file_path  = 'Your filelist path'
mean_std_txt = os.path.join(os.path.join(path, 'mean_std'), 'mean_std.txt')

### Save filelists
metadata={}

with open(file_path, 'r') as fid:
    for line in fid.readlines():
        wav_path, text, spk = line.strip('\n').split("|")
        emo = wav_path.split('/')[-1][0] # Ex) 'a'
        
        clean_char = basic_cleaners(text.rstrip())

        metadata[wav_path] = {'phone':clean_char, 'spk': spk, 'emo': emo}

### Define functions
def text2seq(text):
    sequence=[symbol_to_id['^']]
    sequence.extend(text_to_sequence(text, hparams.text_cleaners))
    sequence.append(symbol_to_id['~'])
    return sequence

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
    P, M = phoneme_count, mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)

def get_mel(filename):
    wav, sr = librosa.load(filename, sr=hparams.sampling_rate)
    wav = librosa.effects.trim(wav, top_db=23, frame_length=1024, hop_length=256)[0]
    wav_32 = wav.astype(np.float32)
    wav = torch.FloatTensor(wav.astype(np.float32))
    melspec, _ = stft.mel_spectrogram(wav.unsqueeze(0))
    return melspec.squeeze(0), wav, wav_32

def get_wav(filename):
    wav, sr = librosa.load(filename, sr=hparams.sampling_rate)
    wav = librosa.effects.trim(wav, top_db=23, frame_length=1024, hop_length=256)[0]
    wav_32 = wav.astype(np.float32)
    return wav_32

def compute_mean_f0(fname, current_spk, current_emo, mean_list, std_list):
    spk_id = metadata[fname]['spk']
    emo_id = metadata[fname]['emo']

    if spk_id == current_spk and emo_id == current_emo:
        wav_32 = get_wav(fname)

        f0, _ = pw.harvest(wav_32.astype(np.float64), hparams.sampling_rate, frame_period=hparams.hop_length/hparams.sampling_rate*1000)

        nonzero_f0 = np.array([x for x in f0 if x!=0]) # Collect only voiced region

        mean_f0 = nonzero_f0.mean()
        std_f0 = nonzero_f0.std()

        mean_list.append(mean_f0)
        std_list.append(std_f0)
    
    return mean_list, std_list

def get_norm_f0(f0, mean, std):
    out = [(x-mean)/std if x!=0 else x for x in f0]
    return out

def save_file(fname):
    phone_seq = torch.LongTensor(text2seq(metadata[fname]['phone']))
    spk_id = metadata[fname]['spk']
    emo_id = metadata[fname]['emo']
    
    melspec, wav, wav_32 = get_mel(fname)

    f0, _ = pw.harvest(wav_32.astype(np.float64), hparams.sampling_rate, frame_period=hparams.hop_length/hparams.sampling_rate*1000)

    name_mean = 'mean_' + spk_id + '_' + emo_id
    name_std = 'std_' + spk_id + '_' + emo_id

    mean_f0 = np.load(os.path.join(mean_std_dir, name_mean + '.npy'))
    std_f0  = np.load(os.path.join(mean_std_dir, name_std + '.npy'))

    norm_f0 = get_norm_f0(f0, mean_f0, std_f0)

    attn_prior = beta_binomial_prior_distribution(len(phone_seq), melspec.size(1), 1)
    
    wav_name = fname.split('/')[-1][:-4]
    np.save(os.path.join(os.path.join(path, 'mels'), wav_name), melspec)
    np.save(os.path.join(os.path.join(path, 'texts'), wav_name), phone_seq)
    np.save(os.path.join(os.path.join(path, 'f0'), wav_name), norm_f0)
    np.save(os.path.join(os.path.join(path, 'alignment_priors'), wav_name), attn_prior) 
    
    return wav_name

##### Preprocessing Start #####
### Search the number of speakers and emotions ###
name, speaker, emotion = process_meta(file_path)
sid_dict = create_id_table(speaker)
eid_dict = create_id_table(emotion)

spk_label = [key for key in sid_dict]
emo_label = [key for key in eid_dict]
print(spk_label)
print(emo_label)

### Compute mean and std of f0 ###
dist_save_txt = open(mean_std_txt, 'w')

for spk in spk_label:
    for emo in emo_label:
        mean_list = [] 
        std_list  = []
        for filepath in tqdm(metadata.keys(), desc=f'{spk}|{emo}'):
            mean_list, std_list = compute_mean_f0(filepath, spk, emo, mean_list, std_list)
        
        mean_list, std_list = np.array([mean_list]), np.array([std_list])
        mean, std = mean_list.mean(), std_list.mean()
        std_of_mean = mean_list.std()

        name_mean = 'mean_' + spk + '_' + emo
        name_std  = 'std_'  + spk + '_' + emo

        np.save(os.path.join(os.path.join(path, 'mean_std'), name_mean), mean)
        np.save(os.path.join(os.path.join(path, 'mean_std'), name_std), std)

        print(f'{spk} - {emo}: mean={mean:.2f}, std={std:.2f}, std of mean={std_of_mean:.2f}')
        dist_save_txt.write(f'{spk} - {emo}: mean={mean:.2f}, std={std:.2f}, std of mean={std_of_mean:.2f}\n')

dist_save_txt.close()

### Prepare Data with Normalized F0 ###
for filepath in tqdm(metadata.keys(), desc='Data preprocessing'):
    _ = save_file(filepath)
   
print('Data preprocessing done!!!')

