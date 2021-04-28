import os,glob
from scipy.io import savemat
import numpy as np

root_path = '/home/nas/user/kbh/CHiME4/WAV/'

#list_SNR = [x for x in glob.glob(os.path.join(root_path,'*')) if os.path.isdir(x)]

data = {"data":{"SNRm5":[
    x for x in glob.glob(os.path.join(root_path,'SNR-5','noisy',"*.wav"))
],"SNR0":[
    x for x in glob.glob(os.path.join(root_path,'SNR0','noisy',"*.wav"))
],"SNRp5":[
    x for x in glob.glob(os.path.join(root_path,'SNR5','noisy',"*.wav"))
],"SNRp10":[
    x for x in glob.glob(os.path.join(root_path,'SNR10','noisy',"*.wav"))
]}}

savemat("list.mat",data)