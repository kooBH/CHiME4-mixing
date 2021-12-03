'''
Evaluation data of CHiME4 are separated by each channel
in CHiME4\data\audio\16kHz\isolated
e.g ) F01_22GC010A_BUS.CH0.wav ~ F01_22GC010A_BUS.CH6.wav
CH0 : close talk mic
CH1 ~ CH6 : tablet multi-channel mic

'''

# generic
import os, glob
# process
import numpy as np
import librosa
import soundfile as sf

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

root_in = '/home/data/kbh/CHiME4/isolated/'
root_out = '/home/data2/kbh/CHiME4_eval/'

list_dir = [
    'dt05_bus_real',
    'dt05_caf_real',
    'dt05_ped_real',
    'dt05_str_real',
    'dt05_bus_simu',
    'dt05_caf_simu',
    'dt05_ped_simu',
    'dt05_str_simu',
    'et05_bus_real',
    'et05_caf_real',
    'et05_ped_real',
    'et05_str_real',
    'et05_bus_simu',
    'et05_caf_simu',
    'et05_ped_simu',
    'et05_str_simu'
]

list_target=[]
for i in range(len(list_dir)):
    list_target = list_target + [x for x in glob.glob(os.path.join(root_in,list_dir[i],'*.CH1.wav'))]

print(len(list_target))

def merge(idx):
    path_1 = list_target[idx]
    path_core = path_1.split('.')[0]
    id = path_core.split('/')[-1]
    dir = path_core.split('/')[-2]
    path_2 = path_core + ".CH2.wav"
    path_3 = path_core + ".CH3.wav"
    path_4 = path_core + ".CH4.wav"
    path_5 = path_core + ".CH5.wav"
    path_6 = path_core + ".CH6.wav"

    raw_1,_ = librosa.load(path_1,sr=16000)
    raw_2,_ = librosa.load(path_2,sr=16000)
    raw_3,_ = librosa.load(path_3,sr=16000)
    raw_4,_ = librosa.load(path_4,sr=16000)
    raw_5,_ = librosa.load(path_5,sr=16000)
    raw_6,_ = librosa.load(path_6,sr=16000)

    data = np.stack((raw_1,raw_2,raw_3,raw_4,raw_5,raw_6),axis=1)

    sf.write(os.path.join(root_out,dir,id+'.wav'),data,16000)

if __name__=='__main__' : 
    
    for i in list_dir : 
        os.makedirs(os.path.join(root_out,i),exist_ok=True)
 
    cpu_num = cpu_count()
    # save 8 threads for others
    cpu_num = cpu_num - 8

    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(merge, arr), total=len(arr),ascii=True,desc='Processing_simu'))

