#!/bin/bash


root=/home/data/kbh/CHiME4/isolated_ext
out=/home/data/kbh/CHiME4/merged_WAV/clean/
list_bus=$(cat list_org_bus)
for value in ${list_bus[@]};do
    echo ${value}
    id=${value:0:16}
    sox -M -c 1 $root/tr05_bus_simu/$id.CH1.Clean.wav -c 1 $root/tr05_bus_simu/$id.CH2.Clean.wav -c 1 $root/tr05_bus_simu/$id.CH3.Clean.wav -c 1 $root/tr05_bus_simu/$id.CH4.Clean.wav -c 1 $root/tr05_bus_simu/$id.CH5.Clean.wav -c 1 $root/tr05_bus_simu/$id.CH6.Clean.wav $out/${id:0:12}.wav
done
list_str=$(cat list_org_str)
for value in ${list_str[@]};do
    echo ${value}
    id=${value:0:16}
    sox -M -c 1 $root/tr05_str_simu/$id.CH1.Clean.wav -c 1 $root/tr05_str_simu/$id.CH2.Clean.wav -c 1 $root/tr05_str_simu/$id.CH3.Clean.wav -c 1 $root/tr05_str_simu/$id.CH4.Clean.wav -c 1 $root/tr05_str_simu/$id.CH5.Clean.wav -c 1 $root/tr05_str_simu/$id.CH6.Clean.wav $out/${id:0:12}.wav
done
list_caf=$(cat list_org_caf)
for value in ${list_caf[@]};do
    echo ${value}
    id=${value:0:16}
    sox -M -c 1 $root/tr05_caf_simu/$id.CH1.Clean.wav -c 1 $root/tr05_caf_simu/$id.CH2.Clean.wav -c 1 $root/tr05_caf_simu/$id.CH3.Clean.wav -c 1 $root/tr05_caf_simu/$id.CH4.Clean.wav -c 1 $root/tr05_caf_simu/$id.CH5.Clean.wav -c 1 $root/tr05_caf_simu/$id.CH6.Clean.wav $out/${id:0:12}.wav
done
list_ped=$(cat list_org_ped)
for value in ${list_ped[@]};do
    echo ${value}
    id=${value:0:16}
    sox -M -c 1 $root/tr05_ped_simu/$id.CH1.Clean.wav -c 1 $root/tr05_ped_simu/$id.CH2.Clean.wav -c 1 $root/tr05_ped_simu/$id.CH3.Clean.wav -c 1 $root/tr05_ped_simu/$id.CH4.Clean.wav -c 1 $root/tr05_ped_simu/$id.CH5.Clean.wav -c 1 $root/tr05_ped_simu/$id.CH6.Clean.wav $out/${id:0:12}.wav
done