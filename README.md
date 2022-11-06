# FluentTTS: Text-dependent Fine-grained Style Control for Multi-style TTS

Official PyTorch Implementation of [FluentTTS: Text-dependent Fine-grained Style Control for Multi-style TTS](https://www.isca-speech.org/archive/pdfs/interspeech_2022/kim22j_interspeech.pdf).
Codes are based on the [Acknowledgements](https://github.com/monglechap/fluenttts#acknowledgements) below.

**Abstract**: In this paper, we propose a method to flexibly control the local prosodic variation of a neural text-to-speech (TTS) model. To provide expressiveness for synthesized speech, conventional TTS models utilize utterance-wise global style embeddings that are obtained by compressing frame-level embeddings along the time axis. However, since utterance-wise global features do not contain sufficient information to represent the characteristics of word-level local features, they are not appropriate for direct use on controlling prosody at a fine scale.
In multi-style TTS models, it is very important to have the capability to control local prosody because it plays a key role in finding the most appropriate text-to-speech pair among many one-to-many mapping candidates.
To explicitly present local prosodic characteristics to the contextual information of the corresponding input text, we propose a module to predict the fundamental frequency ( $F0$ ) of each text by conditioning on the utterance-wise global style embedding.
We also estimate multi-style embeddings using a multi-style encoder, which takes as inputs both a global utterance-wise embedding and a local $F0$ embedding.
Our multi-style embedding enhances the naturalness and expressiveness of synthesized speech and is able to control prosody styles at the word-level or phoneme-level.

Visit our [Demo](https://kchap0118.github.io/fluenttts/) for audio samples.

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

   1-1. Our codes are used for internal Korean dataset. If you run the code with another languages, please modify files in [text](text/) and [hparams.py](hparams.py) that are related to symbols and text preprocessing.

   1-2. Make data filelists like format of [filelists/example_filelist.txt](filelists/example_filelist.txt). They used for preprocessing and training.

   ```
   /your/data/path/angry_f_1234.wav|your_data_text|speaker_type
   /your/data/path/happy_m_5678.wav|your_data_text|speaker_type
   /your/data/path/sadness_f_111.wav|your_data_test|speaker_type
   ...
   ```

   1-3. For finding the number of speaker and emotion and defining file names to save, we used format of [filelists/example_filelist.txt](filelists/example_filelist.txt). Thus, please modify the data-specific part (annotated) in [utils/data_utils.py](utils/data_utils.py), [extract_emb.py](extract_emb.py), [mean_i2i.py](mean_i2i.py) and [inference.py](inference.py)
2. Preprocessing

   2-1. Before run [preprocess.py](preprocess.py), modify path (data path) and file_path (filelist that you make in _1-2_.) in the line [21](https://github.com/monglechap/fluenttts/blob/main/preprocess.py#L21) , [25](https://github.com/monglechap/fluenttts/blob/main/preprocess.py#L25).

   2-2. Run

   ```
   python preprocess.py
   ```

   2-3. Modify path of data, train and validation filelist [hparams.py](hparams.py)

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
   -m: Set the farthest emotion as only neutral or other emotions (explained in mean_i2i.py)
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

## Acknowledgements

We refered to the following codes for official version of implementation.

1. NVIDIA/tacotron2: [Link](https://github.com/NVIDIA/tacotron2)
2. Deepest-Project/Transformer-TTS: [Link](https://github.com/Deepest-Project/Transformer-TTS)
3. NVIDIA/FastPitch: [Link](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)
4. KevinMIN95/StyleSpeech: [Link](https://github.com/KevinMIN95/StyleSpeech)
5. Kyubong/g2pK: [Link](https://github.com/Kyubyong/g2pK)
6. jik876/hifi-gan: [Link](https://github.com/jik876/hifi-gan)
7. KinglittleQ/GST-Tacotron: [Link](https://github.com/KinglittleQ/GST-Tacotron)
