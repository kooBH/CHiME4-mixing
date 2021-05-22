# generic
import os, glob
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

ref_root = '/home/data/kbh/CHiME4/isolated/'
input_root = '/home/data/kbh/CHiME4/CGMM_RLS_MPDR/'
clean_root = '/home/data/kbh/CHiME4/merged_WAV/'
output_root = '/home/data/kbh/MCFE/MFCC/'

list_SNR = ['SNR-7','SNR-5','SNR0','SNR5','SNR7','SNR10']
list_sub = ['noisy','noise','estim','clean']

list_test = [x for x in glob.glob(os.path.join(input_root,'test','clean','*.wav'))]
list_train = [x for x in glob.glob(os.path.join(input_root,'train','clean','*.wav'))]

def convert_test(idx): 
    convert(list_test[idx],'test')

def convert_train(idx):
    convert(list_train[idx],'train')

def convert(path,category):
    target_wav = path.split('/')[-1]
    target_id = target_wav.split('.')[0]

    for snr in list_SNR :
        clean, sr = librosa.load(os.path.join(clean_root,category,snr,'clean',target_id+'.wav'),sr=16000)
        noisy, sr = librosa.load(os.path.join(input_root,category,snr,'noisy',target_id+'.wav'),sr=16000)
        estim, sr = librosa.load(os.path.join(input_root,category,snr,'estimated_speech',target_id+'.wav'),sr=16000)
        noise, sr = librosa.load(os.path.join(input_root,category,snr,'estimated_noise',target_id+'.wav'),sr=16000)

        torch_estim = torch.from_numpy(estim)
        torch_noisy = torch.from_numpy(noisy)
        torch_noise = torch.from_numpy(noise)
        torch_clean = torch.from_numpy(clean)

        torch_estim = torch.unsqueeze(torch_estim,0)
        torch_noisy = torch.unsqueeze(torch_noisy,0)
        torch_noise = torch.unsqueeze(torch_noise,0)
        torch_clean = torch.unsqueeze(torch_clean,0)

        # waveform = [c,n]
        # https://pytorch.org/audio/stable/compliance.kaldi.html
        mfcc_estim = torchaudio.compliance.kaldi.mfcc(waveform = torch_estim)
        mfcc_noisy = torchaudio.compliance.kaldi.mfcc(waveform = torch_noisy)
        mfcc_noise = torchaudio.compliance.kaldi.mfcc(waveform = torch_noise)
        mfcc_clean = torchaudio.compliance.kaldi.mfcc(waveform = torch_clean)

        # save
        torch.save(mfcc_estim,os.path.join(output_root,'MFCC',category,snr,'estim',target_id+'.pt'))
        torch.save(mfcc_noisy,os.path.join(output_root,'MFCC',category,snr,'noisy',target_id+'.pt'))
        torch.save(mfcc_noise,os.path.join(output_root,'MFCC',category,snr,'noise',target_id+'.pt'))
        torch.save(mfcc_clean,os.path.join(output_root,'MFCC',category,snr,'clean',target_id+'.pt'))

if __name__=='__main__' : 
    cpu_num = cpu_count()

    for i in list_SNR :
        for j in list_sub : 
            os.makedirs(os.path.join(output_root,'MFCC','test',i,j),exist_ok=True)
            os.makedirs(os.path.join(output_root,'MFCC','train',i,j),exist_ok=True)

    arr = list(range(len(list_test)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(convert_test, arr), total=len(arr),ascii=True,desc='MFCC test'))


    arr = list(range(len(list_train)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(convert_train, arr), total=len(arr),ascii=True,desc='MFCC train'))
