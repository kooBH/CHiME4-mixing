# generic
import os, glob
import argparse
# process
import numpy as np
import torch
import torchaudio
import librosa
import scipy
import scipy.io
import soundfile

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

fft_size = 1024
n_mels = 40

# path
input_root  = '/home/data/kbh/MCSE/'
## TODO wanna maintain directory structure of files 
output_root = '/home/data/kbh/MCFE/'

# list

list_target = [x for x in glob.glob(os.path.join(  ))]

def convert(idx):
    # https://pytorch.org/audio/stable/transforms.html#melscale
    torchaudio.transforms.MelScale(
        n_mels = n_mels,
        sample_rate = 16000,
        

    )


if __name__=='__main__' : 
    cpu_num = cpu_count()

    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(convert, arr), total=len(arr),ascii=True,desc='stft2mel'+str(n_mels)+' test'))