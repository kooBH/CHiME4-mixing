# generic
import os, glob
# process
import numpy as np
import torch
import torchaudio
import librosa
import scipy
import scipy.io
import soundfile as sf
# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

fft_size = 1024
window = torch.hann_window(window_length=fft_size,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)

input_root =  '/home/data/kbh/CHiME4/CGMM_RLS_MPDR/'
output_root = '/home/data/kbh/MCSE/CGMM_RLS_MPDR/'

list_SNR = ['SNR10','SNR7','SNR5','SNR0','SNR-5','SNR-7']
list_type = ['noisy','estimated_speech','estimated_noise']

list_input = [x for x in glob.glob(os.path.join(input_root,'SNR0','noisy','*.wav'))]

def to_pt(idx):
    file_path = list_input[idx]
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]

    clean_path  = input_root + 'clean_1ch' + '/' + file_name + '.wav'
    clean_out_path =  output_root + 'clean' + '/' + file_name + '.pt'

    clean,_ = librosa.load(clean_path,sr=16000)
    spec_clean = librosa.stft(clean,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)
    spec_clean = np.concatenate((np.expand_dims(spec_clean.real,-1),np.expand_dims(spec_clean.imag,-1)),2)
    torch_clean = torch.from_numpy(spec_clean)
    torch.save(torch_clean,clean_out_path)

    for i in list_SNR :
        noisy_path = input_root + i + '/' + 'noisy' + '/' + file_name + '.wav'
        speech_path = input_root + i + '/' + 'estimated_speech' + '/' + file_name + '.wav'
        noise_path  = input_root + i + '/' + 'estimated_noise' + '/' + file_name + '.wav'

        noisy_out_path =  output_root +  i +'/'+ 'noisy'+ '/'+ file_name + '.pt'
        speech_out_path = output_root +  i +'/'+ 'estimated_speech'+ '/'+ file_name + '.pt'
        noise_out_path = output_root +  i +'/'+ 'estimated_noise'+ '/'+ file_name + '.pt'

        noisy,_ = librosa.load(noisy_path,sr=16000)
        speech,_ = librosa.load(speech_path,sr=16000)
        noise,_ = librosa.load(noise_path,sr=16000)
        
        # noisy is 768 samples longer
        noisy = noisy[:-768]

        spec_noisy = librosa.stft(noisy,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)
        spec_speech = librosa.stft(speech,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)
        spec_noise = librosa.stft(noise,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)

        spec_noisy = np.concatenate((np.expand_dims(spec_noisy.real,-1),np.expand_dims(spec_noisy.imag,-1)),2)
        spec_speech = np.concatenate((np.expand_dims(spec_speech.real,-1),np.expand_dims(spec_speech.imag,-1)),2)
        spec_noise = np.concatenate((np.expand_dims(spec_noise.real,-1),np.expand_dims(spec_noise.imag,-1)),2)

        torch_noisy = torch.from_numpy(spec_noisy)
        torch_speech = torch.from_numpy(spec_speech)
        torch_noise = torch.from_numpy(spec_noise)

        torch.save(torch_noisy,noisy_out_path)
        torch.save(torch_speech,speech_out_path)
        torch.save(torch_noise,noise_out_path)

        #print(torch_noisy.shape)
        #print(torch_speech.shape)
        #print(torch_noise.shape)

if __name__=='__main__':
    os.makedirs(os.path.join(output_root,'clean'),exist_ok=True)
    for i in list_SNR :
        for j in list_type : 
            os.makedirs(os.path.join(output_root,i,j),exist_ok=True)
    
    cpu_num = cpu_count()
    # save 8 threads for others
    cpu_num = 32

#    to_pt(0)
    arr = list(range(len(list_input)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(to_pt, arr), total=len(arr),ascii=True,desc='to_pt'))
