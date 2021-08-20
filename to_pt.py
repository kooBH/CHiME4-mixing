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
overlap_size = fft_size - int(fft_size/4)
window = torch.hann_window(window_length=fft_size,periodic=True, dtype=None, 
                           layout=torch.strided, device=None, requires_grad=False)

#target =  'CGMM_RLS_MPDR'
#target = 'AuxIVA_DC_SVE'
target = 'CGMM_RLS_MPDR_norm_2'

input_root =  '/home/data/kbh/CHiME4/' + target + '/'
wav_root   =  '/home/data/kbh/CHiME4/merged_WAV/'

output_root = '/home/data/kbh/MCSE/' + target + '/'

list_SNR = ['SNR7','SNR5','SNR0','SNR-5','SNR-7']
list_type = ['noisy','estimated_speech','estimated_noise','clean']

list_input = [x for x in glob.glob(os.path.join(input_root,'SNR0','noisy','*.wav'))]

def to_pt(idx):
    file_path = list_input[idx]
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.')[0]

    for i in list_SNR :
        noisy_path = input_root + i + '/' + 'noisy' + '/' + file_name + '.wav'
        speech_path = input_root + i + '/' + 'estimated_speech' + '/' + file_name + '.wav'
        noise_path  = input_root + i + '/' + 'estimated_noise' + '/' + file_name + '.wav'
        clean_path  = wav_root + i + '/' + 'clean' + '/' + file_name + '.wav'

        noisy_out_path =  output_root +  i +'/'+ 'noisy'+ '/'+ file_name + '.pt'
        speech_out_path = output_root +  i +'/'+ 'estimated_speech'+ '/'+ file_name + '.pt'
        noise_out_path = output_root +  i +'/'+ 'estimated_noise'+ '/'+ file_name + '.pt'
        clean_out_path = output_root + i +'/' + 'clean' + '/' + file_name+ '.pt'

        noisy,_ = librosa.load(noisy_path,sr=16000)
        speech,_ = librosa.load(speech_path,sr=16000)
        noise,_ = librosa.load(noise_path,sr=16000)
        clean,_ = librosa.load(clean_path,sr=16000)

        if target == 'CGMM_RLS_MPDR':
            noisy = noisy[:-overlap_size]
        elif target == 'AuxIVA_DC_SVE':
            noisy = noisy[:-overlap_size]
            speech = speech[:-overlap_size]
            noise = noise[:-overlap_size]
            clean = clean[overlap_size:]
        elif target =='CGMM_RLS_MPDR_norm_2':
            clean = clean[:-overlap_size]
            noisy = noisy[overlap_size:]
            speech = speech[overlap_size:]
            noise = noise[overlap_size:]

        spec_noisy = librosa.stft(noisy,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)
        spec_speech = librosa.stft(speech,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)
        spec_noise = librosa.stft(noise,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)
        spec_clean = librosa.stft(clean,window='hann', n_fft=fft_size,hop_length=None, win_length=None,center=False)

        spec_noisy = np.concatenate((np.expand_dims(spec_noisy.real,-1),np.expand_dims(spec_noisy.imag,-1)),2)
        spec_speech = np.concatenate((np.expand_dims(spec_speech.real,-1),np.expand_dims(spec_speech.imag,-1)),2)
        spec_noise = np.concatenate((np.expand_dims(spec_noise.real,-1),np.expand_dims(spec_noise.imag,-1)),2)
        spec_clean= np.concatenate((np.expand_dims(spec_clean.real,-1),np.expand_dims(spec_clean.imag,-1)),2)

        torch_noisy = torch.from_numpy(spec_noisy)
        torch_speech = torch.from_numpy(spec_speech)
        torch_noise = torch.from_numpy(spec_noise)
        torch_clean = torch.from_numpy(spec_noise)

        torch.save(torch_noisy,noisy_out_path)
        torch.save(torch_speech,speech_out_path)
        torch.save(torch_noise,noise_out_path)
        torch.save(torch_clean,clean_out_path)

        #print(torch_noisy.shape)
        #print(torch_speech.shape)
        #print(torch_noise.shape)

if __name__=='__main__':
    for i in list_SNR :
        for j in list_type : 
            os.makedirs(os.path.join(output_root,i,j),exist_ok=True)
    
    print(len(list_input))

    cpu_num = cpu_count()
    # save 8 threads for others
    cpu_num = 66

    arr = list(range(len(list_input)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(to_pt, arr), total=len(arr),ascii=True,desc='to_pt'))
