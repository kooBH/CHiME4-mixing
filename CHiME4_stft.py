import os, glob
import argparse
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

# param
parser = argparse.ArgumentParser()
parser.add_argument('--fft_size', '-n', type=int, required=False, default=1024)
parser.add_argument('--output_root', '-o', type=str, required=True)
args = parser.parse_args()

fft_size = args.fft_size

## ROOT
noisy_root = '/home/data/kbh/CHiME4/isolated_1ch_track/'
estim_root = '/home/data/kbh/CGMM_RLS_MPDR/trial_04/'
mask_root = '/home/data/kbh/CGMM_RLS_MPDR/trial_04_mask/'
output_root = args.output_root

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
        raise Exception('np.shape(noisy)'+str(np.shape(noisy))+' != np.shape(estim) ' + str(np.shape(estim)))

    noisy_spec = librosa.stft(noisy,window='hann',n_fft=fft_size,center=False)
    estim_spec = librosa.stft(estim,window='hann',n_fft=fft_size,center=False)
    noise_spec = noisy_spec * mask

    torch_estim = torch.from_numpy(estim_spec)
    torch_noisy = torch.from_numpy(noisy_spec)
    torch_noise = torch.from_numpy(noise_spec)

    # save
    torch.save(torch_estim,os.path.join(output_root,'STFT-'+str(fft_size),'measure','estim',target_category,target_id+'.pt'))
    torch.save(torch_noisy,os.path.join(output_root,'STFT-'+str(fft_size),'measure','noisy',target_category,target_id+'.pt'))
    torch.save(torch_noise,os.path.join(output_root,'STFT-'+str(fft_size),'measure','noise',target_category,target_id+'.pt'))

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
        os.makedirs(os.path.join(output_root,'STFT-'+str(fft_size),'measure','estim',i),exist_ok=True)
        os.makedirs(os.path.join(output_root,'STFT-'+str(fft_size),'measure','noisy',i),exist_ok=True)
        os.makedirs(os.path.join(output_root,'STFT-'+str(fft_size),'measure','noise',i),exist_ok=True)

    arr = list(range(len(noisy_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='CHiME4 STFT-'+str(fft_size)))


