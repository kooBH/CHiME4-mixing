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

# param
parser = argparse.ArgumentParser()
parser.add_argument('--n_mels', '-m', type=int, required=True)
parser.add_argument('--fft_size', '-n', type=int, required=False, default=1024)
args = parser.parse_args()

fft_size = args.fft_size
if fft_size%4 != 0 :
    raise Exception(fft_size%4 != 0)
shift_size = int(fft_size/4)
n_mels = args.n_mels

# mel
mel_basis = librosa.filters.mel(sr=16000, n_fft=fft_size, n_mels=n_mels)

# path and dir
ref_root = '/home/data/kbh/CHiME4/isolated/'
input_root = '/home/data/kbh/CHiME4/CGMM_RLS_MPDR/'
clean_root = '/home/data/kbh/CHiME4/merged_WAV/'
#output_root = '/home/data/kbh/MCFE/MFCC/'
output_root = '/home/kbh/kiosk/MCFE/CGMM_RLS_MPDR/'

list_SNR = ['SNR-7','SNR-5','SNR0','SNR5','SNR7','SNR10']
list_sub = ['noisy','noise','estim','clean']

list_test = [x for x in glob.glob(os.path.join(input_root,'test','clean','*.wav'))]
list_train = [x for x in glob.glob(os.path.join(input_root,'train','clean','*.wav'))]

def convert_test(idx): 
    convert(list_test[idx],'test')

def convert_train(idx):
    convert(list_train[idx],'train')

def convert(path,category):
    sr = 16000
    target_wav = path.split('/')[-1]
    target_id = target_wav.split('.')[0]

    for snr in list_SNR :
        clean, sr = librosa.load(os.path.join(clean_root,snr,'clean',target_id+'.wav'),sr=sr)
        noisy, sr = librosa.load(os.path.join(input_root,category,snr,'noisy',target_id+'.wav'),sr=sr)
        # adjust mistake in mixing

        if target =='CGMM_RLS_MPDR':
            noisy = noisy[:-768]

        estim, sr = librosa.load(os.path.join(input_root,category,snr,'estimated_speech',target_id+'.wav'),sr=sr)
        noise, sr = librosa.load(os.path.join(input_root,category,snr,'estimated_noise',target_id+'.wav'),sr=sr)

        noisy_spec = librosa.stft(noisy,window='hann', n_fft=fft_size,hop_length=shift_size, win_length=None,center=False)
        estim_spec = librosa.stft(estim,window='hann', n_fft=fft_size,hop_length=shift_size, win_length=None,center=False)
        noise_spec = librosa.stft(noise,window='hann', n_fft=fft_size,hop_length=shift_size, win_length=None,center=False)
        clean_spec = librosa.stft(clean,window='hann', n_fft=fft_size,hop_length=shift_size, win_length=None,center=False)

        noisy_mel = np.matmul(mel_basis,np.abs(noisy_spec))
        estim_mel = np.matmul(mel_basis,np.abs(estim_spec))
        noise_mel = np.matmul(mel_basis,np.abs(noise_spec))
        clean_mel = np.matmul(mel_basis,np.abs(clean_spec))

        torch_noisy = torch.from_numpy(noisy_mel)
        torch_estim = torch.from_numpy(estim_mel)
        torch_noise = torch.from_numpy(noise_mel)
        torch_clean = torch.from_numpy(clean_mel)

        # save
        torch.save(torch_noisy,os.path.join(output_root,'mel-'+str(n_mels),category,snr,'noisy',target_id+'.pt'))
        torch.save(torch_estim,os.path.join(output_root,'mel-'+str(n_mels),category,snr,'estim',target_id+'.pt'))
        torch.save(torch_noise,os.path.join(output_root,'mel-'+str(n_mels),category,snr,'noise',target_id+'.pt'))
        torch.save(torch_clean,os.path.join(output_root,'mel-'+str(n_mels),category,snr,'clean',target_id+'.pt'))

if __name__=='__main__' : 
    cpu_num = cpu_count()

    for i in list_SNR :
        for j in list_sub : 
            os.makedirs(os.path.join(output_root,'mel-'+str(n_mels),'test',i,j),exist_ok=True)
            os.makedirs(os.path.join(output_root,'mel-'+str(n_mels),'train',i,j),exist_ok=True)

    arr = list(range(len(list_test)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(convert_test, arr), total=len(arr),ascii=True,desc='mel'+str(n_mels)+' test'))


    arr = list(range(len(list_train)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(convert_train, arr), total=len(arr),ascii=True,desc='mel-'+str(n_mels)+' train'))
