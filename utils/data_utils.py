import os, random, torch, pdb
import numpy as np
import torch.utils.data
import torch.nn.functional as F

import hparams


# Prepare filelists
def process_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        name, speaker, emotion = [], [], []
        for line in f.readlines():
            # Data-specific configuration. 
            # You should change this codes for your own data structure
            path, text, spk = line.strip('\n').split('|')
            filename = path.split('/')[-1][:-4]
            emo = filename[0]

            name.append(filename)
            speaker.append(spk)
            emotion.append(emo)

        return name, speaker, emotion

# Sort unique IDs
def create_id_table(ids):
    sorted_ids = np.sort(np.unique(ids))
    d = {sorted_ids[i]: i for i in range(len(sorted_ids))}
    return d

# Read filelists
def load_filepaths_and_text(metadata, split="|"):
    with open(metadata, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

# Custom dataset
class TextMelSet(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.seq_dir     = os.path.join(hparams.data_path, 'texts')
        self.mel_dir     = os.path.join(hparams.data_path, 'mels')
        self.norm_f0_dir = os.path.join(hparams.data_path, 'pitch_norm')
        self.prior_dir   = os.path.join(hparams.data_path, 'alignment_priors')

        _, self.speaker, self.emotion = process_meta(audiopaths_and_text)
        self.sid_dict = create_id_table(self.speaker)
        self.eid_dict = create_id_table(self.emotion)
#        print(self.sid_dict)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        # Data-specific configuration. 
        # You should change this codes for your own data structure
        wav_path = audiopath_and_text[0]
        name = wav_path.split('/')[-1][:-4]
        emo = name[0]
        spk = audiopath_and_text[2]

        spk_id = self.sid_dict[spk]
        emo_id = self.eid_dict[emo]
        
        text   = np.load(os.path.join(self.seq_dir, name+'.npy'))
        mel    = np.load(os.path.join(self.mel_dir, name+'.npy'))
        f0     = np.load(os.path.join(self.norm_f0_dir, name+'.npy'))
        prior  = np.load(os.path.join(self.prior_dir, name+'.npy'))
        
        return (torch.IntTensor(text), torch.FloatTensor(mel), torch.FloatTensor(f0), 
                torch.FloatTensor(prior), name, torch.LongTensor([spk_id]), emo)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

# Collate function
class TextMelCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]),
                                                          dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include Spec padded and gate padded
        mel_padded     = torch.zeros(len(batch), num_mels, max_target_len)
        gate_padded    = torch.zeros(len(batch), max_target_len)
        f0_padded      = torch.zeros(len(batch), max_target_len)
        prior_padded   = torch.zeros(len(batch), max_input_len, max_target_len)

        output_lengths = torch.LongTensor(len(batch))
        name = []
        spk  = torch.LongTensor(len(batch))
        emo  = []

        for i in range(len(ids_sorted_decreasing)):
            mel   = batch[ids_sorted_decreasing[i]][1]
            f0    = batch[ids_sorted_decreasing[i]][2]
            prior = batch[ids_sorted_decreasing[i]][3].contiguous().transpose(0,1)

            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            f0_padded[i, :mel.size(1)]     = f0
        
            prior_padded[i, :prior.size(0), :prior.size(1)] = prior

            output_lengths[i] = mel.size(1)
            name.append(batch[ids_sorted_decreasing[i]][4])
            spk[i] = batch[ids_sorted_decreasing[i]][5]
            emo.append(batch[ids_sorted_decreasing[i]][6])
        
        return text_padded, input_lengths, mel_padded, output_lengths, gate_padded, \
               f0_padded, prior_padded, name, spk, emo			   
