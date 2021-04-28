# Mixed audio of CHiME4 -> .pt files

import torch
import os,glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

input_root = '/home/nas/user/kbh/CHiME4/WAV/'
output_root = '/home/nas/user/kbh/CHiME4/PT/'

window_size = 1024
window = torch.hann_window(window_length=fft_size,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)

for SNR in ['SNR-5','SNR0','SNR5','SNR10']:
    os.makedirs(os.path.join(out_root,SNR),exist_ok=True)
    for category in ['clean','noise','noisy'] :
        os.makedirs(os.path.join(out_root,SNR,category),exist_ok=True)

target_list = [x for x in glob.glob(os.path.join(input_root,'*','*','*.wav')]


