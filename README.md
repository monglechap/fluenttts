# FluentTTS: Text-dependent Fine-grained Style Control for Multi-style TTS

Official PyTorch Implementation of [FluentTTS: Text-dependent Fine-grained Style Control for Multi-style TTS](https://www.isca-speech.org/archive/pdfs/interspeech_2022/kim22j_interspeech.pdf).
Codes are based on the reference below.

Visit out [Demo](https://kchap0118.github.io/fluenttts/) for audio samples.

## Prerequisites

- Clone this repository
- Install python requirements. Please refer [requirements.txt](requirements.txt)
- Like [Code reference](https://github.com/Deepest-Project/Transformer-TTS), please modify return values of _torch.nn.funtional.multi_head_attention.forward()_ to draw attention of all head in the validation step.
  ```
  #Before
  return attn_output, attn_output_weights.sum(dim=1) / num_heads
  #After
  return attn_output, attn_output_weights
  ```

## Preprocessing

1. Prepare text preprocessing
   1-1. Our code used for phoneme (Korean) dataset. If you run the code with another languages, please modify files in [text](text/) and [hparams.py](hparams.py) that are related to symbols and text preprocessing.
   1-2. Make data filelists like [filelists/example_filelist.txt](filelists/example_filelist.txt).

   ```
   /your/data/path/angry_f_1234.wav|your_data_text|speaker_type
   /your/data/path/happy_m_5678.wav|your_data_text|speaker_type
   ```
2. Preprocessing

   2-1. Before run [preprocess.py](preprocess.py), modify path (data path) and file_path (filelist that you make in 1-2.) in the line 21, 25.

   2-2. Run

   ```
   python preprocess.py
   ```

## Training

```
python train.py -o [SAVE DIRECTORY PATH] -m [BASE OR PROP] 
```

(Arguments)

```
-c: Ckpt path for loading
-o: Path for saving ckpt and log
-m: Choose baseline or proposed model
```

## Inference

0. Mean (i2i) style embedding extraction (optional)
   0-1. Extract emotion embeddings of dataset

   ```
   python extract_emb.py -o [SAVE DIRECTORY PATH] -c [CHECKPOINT PATH] -m [BASE OR PROP]
   ```

   (Arguments)

   ```
   -o: Path for saving emotion embs
   -c: Ckpt path for loading
   -m: Choose baseline or proposed model
   ```

   0-2. Compute mean (or I2I) embs

   ```
   python mean_i2i.py -i [EXTRACED EMB PATH] -o [SAVE DIRECTORY PATH] -m [NEU OR ALL]
   ```

   (Arguments)

   ```
   -i: Path of saved emotion embs
   -o: Path for saving mean or i2i embs
   -m: Set the farthest emotion as only neutral or other emotions
   ```
1. Inference

   ```
   python inference.py -c [CHECKPOINT PATH] -v [VOCODER PATH] -s [MEAN EMB PATH] -o [SAVE DIRECTORY PATH] -m [BASE OR PROP]
   ```

   (Arguments)

   ```
   -c: Ckpt path of acoustic model
   -v: Ckpt path of vocoder (Hifi-GAN)
   -s (optional): Path of saved mean (i2i) embs
   -o: Path for saving generated wavs
   -m: Choose baseline or proposed model
   --control (optional): F0 controal at the utterance or phoneme-level
   --hz (optional): values to modify F0
   --ref_dir (optional): Path of reference wavs. Use when you do not apply mean (i2i) algs.
   --spk (optional): Use with --ref_dir
   --emo (optional): Use with --ref_dir
   --slide (optional): Use when you want to apply sliding window attn in Multispeech
   ```

# Reference

1. NVIDIA/tacotron2: [Link](https://github.com/NVIDIA/tacotron2)
2. Deepest-Project/Transformer-TTS: [Link](https://github.com/Deepest-Project/Transformer-TTS)
3. NVIDIA/FastPitch: [Link](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)
4. KevinMIN95/StyleSpeech: [Link](https://github.com/KevinMIN95/StyleSpeech)
5. Kyubong/g2pK: [Link](https://github.com/Kyubyong/g2pK)
6. jik876/hifi-gan: [Link](https://github.com/jik876/hifi-gan)
