import numpy as np
import os,glob

import librosa
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

input_root  =  '/home/data/kbh/CHiME4/merged_WAV/clean/'
output_root =  '/home/data/kbh/CHiME4/CGMM_RLS_MPDR/clean_1ch/'

list_input = [x for x in glob.glob(os.path.join(input_root,'*.wav'))]


delay = 1024 - 256
def adjust(idx):
    file_path = list_input[idx]
    file_name = file_path.split('/')[-1]

    x,_ = librosa.load(file_path,mono=False,sr=16000)
    pad = np.zeros(delay)
    y = np.concatenate((pad,x[1,:-768]))

    output_path = output_root + '/' + file_name
    sf.write(output_path,y,16000)

#num_cpu = cpu_count()
num_cpu = 32

arr = list(range(len(list_input)))
with Pool(num_cpu) as p:
    r = list(tqdm(p.imap(adjust, arr), total=len(arr),ascii=True,desc='CHiME4-clean_1ch'))