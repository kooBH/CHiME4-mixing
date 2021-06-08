import os, glob
import scipy.io as sio
import numpy as np
import librosa
import torch
import torchaudio
import scipy.io as sio

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

## ROOT
noisy_root = '/home/data/kbh/CHiME4/isolated_1ch_track/'
estim_root = '/home/data/kbh/CGMM_RLS_MPDR/trial_04/'
mask_root = '/home/data/kbh/CGMM_RLS_MPDR/trial_04_mask/'
output_root = '/home/data/kbh/MCFE/inference/'

#noisy_root = '/home/kiosk/dnn2/CHiME4/isolated_1ch_track/'
#estim_root = '/home/kiosk/dnn2/CGMM_RLS_MPDR/trial_04/'
#mask_root =  '/home/kiosk/dnn2/CGMM_RLS_MPDR/trial_04_mask/'

## PATH
noisy_list = [x for x in glob.glob(os.path.join(noisy_root,'*','*.wav'))]
print(len(noisy_list))

def process(idx):
    target_path = noisy_list[idx]
    target_name = target_path.split('/')[-1]
    target_id = target_name.split('.')[0]
    target_category = target_path.split('/')[-2]

    noisy,_ = librosa.load(target_path,sr=16000)
    estim,_ = librosa.load(os.path.join(estim_root,target_category,target_name),sr=16000)
    mask = sio.loadmat(os.path.join(mask_root,target_category,target_id+'.mat'))['noise_mask']

    if np.shape(noisy) != np.shape(estim) :
        print(target_id)
        raise Exception('np.shape(noisy) != np.shape(estim)')

    noisy_spec = librosa.stft(noisy,window='hann',n_fft=1024,center=False)
    estim_spec = librosa.stft(estim,window='hann',n_fft=1024,center=False)
    noise_spec = noisy_spec * mask

    noisy = librosa.istft(noisy_spec,win_length = 1024,center=False)
    estim = librosa.istft(estim_spec,win_length = 1024,center=False)
    noise = librosa.istft(noise_spec,win_length = 1024,center=False)

    torch_estim = torch.from_numpy(estim)
    torch_noisy = torch.from_numpy(noisy)
    torch_noise = torch.from_numpy(noise)

    torch_estim = torch.unsqueeze(torch_estim,0)
    torch_noisy = torch.unsqueeze(torch_noisy,0)
    torch_noise = torch.unsqueeze(torch_noise,0)

    # waveform = [c,n]
    # https://pytorch.org/audio/stable/compliance.kaldi.html
    mfcc_estim = torchaudio.compliance.kaldi.mfcc(waveform = torch_estim)
    mfcc_noisy = torchaudio.compliance.kaldi.mfcc(waveform = torch_noisy)
    mfcc_noise = torchaudio.compliance.kaldi.mfcc(waveform = torch_noise)

    # save
    torch.save(mfcc_estim,os.path.join(output_root,'MFCC','estim',target_category,target_id+'.pt'))
    torch.save(mfcc_noisy,os.path.join(output_root,'MFCC','noisy',target_category,target_id+'.pt'))
    torch.save(mfcc_noise,os.path.join(output_root,'MFCC','noise',target_category,target_id+'.pt'))


if __name__=='__main__': 
    cpu_num = cpu_count()

    dirs_1 = ['dt','et']
    dirs_2 = ['bus','caf','str','ped']
    dirs_3 = ['real','simu']
    dirs =[]
    for i in dirs_1 :
        for j in dirs_2 :
            for k in dirs_3 :
                dirs.append(i+'05_'+j+'_'+k)
        
    for i in dirs :
        os.makedirs(os.path.join(output_root,'MFCC','estim',i),exist_ok=True)
        os.makedirs(os.path.join(output_root,'MFCC','noisy',i),exist_ok=True)
        os.makedirs(os.path.join(output_root,'MFCC','noise',i),exist_ok=True)

    arr = list(range(len(noisy_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='CHiME4 MFCC'))


