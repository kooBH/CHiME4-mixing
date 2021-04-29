import numpy as np
import os,glob

import librosa
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

root_path = '/home/data/kbh/CHiME4/'
out_root = '/home/data/kbh/CHiME4/merged_WAV/'

# Need to use all 6 channels
list_dt= [x for x in glob.glob(os.path.join(root_path,'isolated/dt05_bth','*.CH1.wav'))]
list_et = [x for x in glob.glob(os.path.join(root_path,'isolated/et05_bth','*.CH1.wav'))]
list_tr = [x for x in glob.glob(os.path.join(root_path,'isolated/tr05_bth','*.CH1.wav'))]
# /home/nas/DB/CHiME4/data/audio/16kHz/backgrounds/BGD_150203_010_PED.CH1.wav'
list_noise = [x for x in glob.glob(os.path.join(root_path,'backgrounds','*.CH1.wav'))]

# NOTE : Merging noise is too slow, use merge_noise.sh

list_audio = list_dt + list_et + list_tr
print(len(list_audio))

def merge(idx) : 
    path = list_audio[idx]
    pre_path = path.split('.')[0]
    id = pre_path.split('/')[-1]

    audio,_ = librosa.load(path,sr=16000)
    audio = np.expand_dims(audio,axis=1)
    # merge all channels
    for str_ch in ['CH2','CH3','CH4','CH5','CH6'] : 
        tmp_path = pre_path + '.' +str_ch + '.' + 'wav'
        tmp_audio,_ = librosa.load(tmp_path,sr=16000)
        tmp_audio = np.expand_dims(tmp_audio,axis=1)
        audio = np.concatenate([audio,tmp_audio],axis=1)
    if pre_path.split('/')[-2]=='backgrounds' : 
        sf.write(os.path.join(out_root,'noise',id+'.wav'),audio,16000)
    else :
        sf.write(os.path.join(out_root,'clean',id+'.wav'),audio,16000)


for category in ['clean','noise'] :
    os.makedirs(os.path.join(out_root,category),exist_ok=True)

    

#num_cpu = cpu_count()
num_cpu = 32

arr = list(range(len(list_audio)))
with Pool(num_cpu) as p:
    r = list(tqdm(p.imap(merge, arr), total=len(arr),ascii=True,desc='CHiME4-Merge'))