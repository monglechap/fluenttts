from text import symbols

### Experiment Parameters
seed = 118
n_gpus = 1
data_path = '/YOUR/PREPROCESSED/DATA/PATH'
training_files = 'filelists/your_train_file.txt'
validation_files = 'filelists/your_valid_file.txt'
inference_files = 'inference_textlist.txt'
text_cleaners = ['basic_cleaners'] # For Korean

### Audio Parameters
sampling_rate = 16000
filter_length = 1024
hop_length = 200 # 12.5ms
win_length = 800 # 50ms
n_mel_channels = 80
mel_fmin = 50
mel_fmax = 7200

### Model Parameters
n_symbols = len(symbols)
symbols_embedding_dim = 256
hidden_dim = 256
spk_hidden_dim = 16
dprenet_dim = 32
ff_dim = 1024
n_heads = 2 
n_layers = 4
sliding_window = [-1, 4] # For sliding window attention in inference

# Multi-style generation
ms_kernel = 3
n_layers_lp_enc = 6

# reference encoder
E = 256
ref_enc_filters = [32,32,64,64,128,128]
ref_enc_size = [3,3]
ref_enc_strides = [2,2]
ref_enc_pad = [1,1]
ref_enc_gru_size = E // 2

# Loss scale
emo_scale = 1.0
f0_scale = 1.0
kl_scale = 0.1

# Dataset configuration
num_spk = 4
num_emo = 4

### Optimization Hyperparameters
lr = 0.05
batch_size = 32
warmup_steps = 4000
grad_clip_thresh = 1.0

iters_per_validation = 5000
iters_per_checkpoint = 10000

training_epochs = 100000
train_steps = 500000
local_style_step = 20000
bin_loss_enable_steps = 10000
bin_loss_warmup_steps = 5000

### HiFi-GAN 
resblock = "1"
num_gpus = 0
learning_rate = 0.0002
adam_b1 = 0.8
adam_b2 = 0.99
lr_decay = 0.999

upsample_rates = [5,5,4,2]
upsample_kernel_sizes = [9,9,8,4]
upsample_initial_channel = 512
resblock_kernel_sizes = [3,7,11]
resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]

segment_size = 6400
num_mels = 80
num_freq = 1025
n_fft = 1024
hop_size = 200
win_size = 800

fmin = 50
fmax = 7200
num_workers = 4

