import numpy as np
import os,glob

import librosa
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

root_path = '/home/nas/DB/CHiME4/data/audio/16kHz/'
out_root = '/home/nas/user/kbh/CHiME4/WAV/'

# Need to use all 6 channels
list_clean = [x for x in glob.glob(os.path.join(root_path,'isolated/tr05_bth','*.CH1.wav'))]
list_clean = list_clean + [x for x in glob.glob(os.path.join(root_path,'isolated/tr05_org','*.wav'))]
# /home/nas/DB/CHiME4/data/audio/16kHz/backgrounds/BGD_150203_010_PED.CH1.wav'
list_noise = [x for x in glob.glob(os.path.join(root_path,'backgrounds','*.CH1.wav'))]
data_noise = []
for n_path in list_noise:
    noise,_ = librosa.load(n_path,sr=16000)
    noise = np.expand_dims(noise,axis=-1)
    # merge all channels
    for str_ch in ['CH2','CH3','CH4','CH5','CH6'] : 
        pre_path = n_path.split('.')[0]
        tmp_path = pre_path + '.' +str_ch + '.' + 'wav'
        tmp_noise,_ = librosa.load(tmp_path,sr=16000)
        tmp_noise = np.expand_dims(tmp_noise,axis=-1)
        noise = np.concatenate([noise,tmp_noise],axis=1)
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

    clean,_ = librosa.load(path_clean,sr=16000)
    clean = np.expand_dims(clean,axis=-1)
    # merge all channels
    for str_ch in ['CH2','CH3','CH4','CH5','CH6'] : 
        pre_path = n_path.split('.')[0]
        tmp_path = pre_path + '.' +str_ch + '.' + 'wav'
        tmp_clean,_ = librosa.load(tmp_path,sr=16000)
        tmp_clean = np.expand_dims(tmp_clean,axis=-1)
        clean = np.concatenate([clean,tmp_clean],axis=1)

    # sampling
    len_clean = len(clean)
    
    # mixing for SNRs
    for SNR in [-5, 0, 5, 10] :
        idx_noise = np.random.randint(num_noise)
        noise = data_noise[idx_noise]
        interval_noise = np.random.randint(len(noise)-len_clean)
        sampled_noise = noise[interval_noise:interval_noise+len_clean]
        clean2,noise2 = setSNR(clean,sampled_noise,SNR)
        noisy = clean2+noise2
        normalization_ratio = np.max(np.abs(noisy))
        noisy = noisy/normalization_ratio
        clean2= clean2/normalization_ratio
        noise2= noise2/normalization_ratio

        sf.write(os.path.join(out_root,'SNR'+str(SNR),'clean',id+'.wav'),clean2,16000)
        sf.write(os.path.join(out_root,'SNR'+str(SNR),'noise',id+'.wav'),noise2,16000)
        sf.write(os.path.join(out_root,'SNR'+str(SNR),'noisy',id+'.wav'),noisy,16000)


for SNR in ['SNR-10','SNR-5','SNR0','SNR5','SNR10']:
    os.makedirs(os.path.join(out_root,SNR),exist_ok=True)
    for category in ['clean','noise','noisy'] :
        os.makedirs(os.path.join(out_root,SNR,category),exist_ok=True)

num_cpu = cpu_count()

arr = list(range(len(list_clean)))
with Pool(num_cpu) as p:
    r = list(tqdm(p.imap(mix, arr), total=len(arr),ascii=True,desc='CHiME4-Mixing'))