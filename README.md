# CHiME4 Data Preparation

## Clean data  
+ 6ch  
dt05_bth + et05_bth +  tr05_bth : 1139   
+ 1ch  
tr05_org : 7138  
 
## Noise data  
+ 6ch  
16 

## SNR  
-7, -5, 0, 5, 7, 10

## Process for train data
1. merge_org.sh : tr05_org simulated data -> 6ch clean data    
2. merge_noise.sh  : isolated noise data -> 6ch noise data  
3. merge_CHiME4.py : isolated clean bth data -> 6ch clean data  
4. mix_CHiME4.py   : mixing, clean+noise with SNRs  
5. preprocess.m    : preprocess,  noisy data  > estimated speech, estimated noise, clean , 1ch each  
6. clean_1ch.py    : extract and sync merged WAV to 1ch clean data    
7. to_pt.py        :  wav -> pytorch data in STFT domain    

## Extra : 
classification.py : mv .pt data orignated from dt,et to another directory   

## Process for test data
+ inference_mfcc.py
+ inference_mel.py


## Extra : MFCC
1. mfcc.py