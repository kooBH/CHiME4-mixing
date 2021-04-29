import numpy as np
import os,glob

import librosa
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

root_path = '/home/data/kbh/CHiME4/merged_WAV/'
out_root = '/home/data/kbh/CHiME4/merged_WAV/'

list_clean = [x for x in glob.glob(os.path.join(root_path,'clean','*.wav'))]
# /home/nas/DB/CHiME4/data/audio/16kHz/backgrounds/BGD_150203_010_PED.CH1.wav'
list_noise = [x for x in glob.glob(os.path.join(root_path,'noise','*.wav'))]

data_noise = []
for n_path in tqdm(list_noise):
    noise,_ = librosa.load(n_path,sr=16000)
    data_noise.append(noise)
num_noise = len(list_noise)

def setSNR(clean,noise,SNR=0,normalized=False):
    if not normalized : 
        clean = clean/np.max(np.abs(clean))
        noise = noise/np.max(np.abs(noise))

    mean_energy_clean = np.sum(np.power(clean,2))
    mean_energy_noise = np.sum(np.power(noise,2))
    energy_normal = np.sqrt(mean_energy_clean)/np.sqrt(mean_energy_noise)
    SNR_weight = energy_normal/np.sqrt(np.power(10,SNR/10))

    if SNR >= 0 :
        # decrease erergy of noise
        noise = noise*SNR_weight
    else :
        # decrease erergy of clean
        clean = clean / SNR_weight
    
    return clean, noise


def mix(idx):
    # load
    path_clean = list_clean[idx]
    id = path_clean.split('/')[-1]
    id = id.split('.')[0]

    clean,_ = librosa.load(path_clean,mono=False,sr=16000)

    # sampling
    len_clean = np.shape(clean)[1]
    
    # mixing for SNRs
    for SNR in [-10,-7,-5, 0, 5,7, 10] :
        idx_noise = np.random.randint(num_noise)
        noise,_ = librosa.load(list_noise[idx_noise],mono=False,sr=16000)
        interval_noise = np.random.randint(np.shape(noise)[1]-len_clean)
        sampled_noise = noise[:,interval_noise:interval_noise+len_clean]
        clean2,noise2 = setSNR(clean,sampled_noise,SNR)
        noisy = clean2+noise2

        # normalization
        normalization_ratio = np.max(np.abs(noisy))
        noisy = noisy/normalization_ratio
        clean2= clean2/normalization_ratio
        noise2= noise2/normalization_ratio

        # saving in (frames,channels)
        noisy = noisy.swapaxes(1,0)
        clean2 = clean2.swapaxes(1,0)
        noise2 = noise2.swapaxes(1,0)

        sf.write(os.path.join(out_root,'SNR'+str(SNR),'clean',id+'.wav'),clean2,16000)
        sf.write(os.path.join(out_root,'SNR'+str(SNR),'noise',id+'.wav'),noise2,16000)
        sf.write(os.path.join(out_root,'SNR'+str(SNR),'noisy',id+'.wav'),noisy,16000)


for SNR in ['SNR-10','SNR-7','SNR-5','SNR0','SNR5','SNR7','SNR10']:
    os.makedirs(os.path.join(out_root,SNR),exist_ok=True)
    for category in ['clean','noise','noisy'] :
        os.makedirs(os.path.join(out_root,SNR,category),exist_ok=True)

num_cpu = cpu_count()

arr = list(range(len(list_clean)))
with Pool(num_cpu) as p:
    r = list(tqdm(p.imap(mix, arr), total=len(arr),ascii=True,desc='CHiME4-Mixing'))