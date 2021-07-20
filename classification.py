import os,glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

ref_root = '/home/data/kbh/CHiME4/isolated/'

#target = 'CGMM_RLS_MPDR'
target = 'AuxIVA_DC_SVE'


work_root = '/home/data/kbh/MCSE/'+target+'/train/'
test_root = '/home/data/kbh/MCSE/'+target+'/test/'

list_SNR = ['SNR-7','SNR-5','SNR0','SNR5','SNR7','SNR10']
list_sub = ['noisy','estimated_speech','estimated_noise']

list_dt = [x for x in glob.glob(os.path.join(ref_root,'dt05_bth','*.CH1.wav'))]
list_et = [x for x in glob.glob(os.path.join(ref_root,'et05_bth','*.CH1.wav'))]


os.makedirs(os.path.join(test_root,'clean'), exist_ok = True)
for i in list_SNR : 
    for j in list_sub : 
        os.makedirs(os.path.join(test_root,i,j),exist_ok = True)

def dt(idx):
    ref_path = list_dt[idx]
    name = ref_path.split('/')[-1]
    name = name.split('.')[0]
    name = name + '.pt'
    shutil.move(os.path.join(work_root,'clean',name),os.path.join(test_root,'clean',name))
    for i in list_SNR : 
        for j in list_sub : 
            shutil.move(os.path.join(work_root,i,j,name),os.path.join(test_root,i,j,name))

def et(idx):
    ref_path = list_et[idx]
    name = ref_path.split('/')[-1]
    name = name.split('.')[0]
    name = name + '.pt'
    shutil.move(os.path.join(work_root,'clean',name),os.path.join(test_root,'clean',name))
    for i in list_SNR : 
        for j in list_sub : 
            shutil.move(os.path.join(work_root,i,j,name),os.path.join(test_root,i,j,name))

#num_cpu = cpu_count()
num_cpu = 64

arr = list(range(len(list_dt)))
with Pool(num_cpu) as p:
    r = list(tqdm(p.imap(dt, arr), total=len(arr),ascii=True,desc='dt05'))

arr = list(range(len(list_et)))
with Pool(num_cpu) as p:
    r = list(tqdm(p.imap(et, arr), total=len(arr),ascii=True,desc='et05'))