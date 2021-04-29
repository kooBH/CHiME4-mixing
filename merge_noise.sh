#!/bin/bash

# Merging in python is too slow
list="BGD_150203_010_PED BGD_150203_010_STR BGD_150203_020_PED BGD_150204_010_BUS BGD_150204_020_BUS BGD_150204_020_CAF BGD_150204_030_BUS BGD_150204_030_CAF BGD_150204_040_BUS BGD_150205_030_PED BGD_150205_040_CAF BGD_150211_020_STR BGD_150211_030_STR BGD_150211_040_PED BGD_150212_040_STR BGD_150212_050_STR"

for value in ${list[@]};do
  sox -M   -c 1 $value.CH1.wav  -c 1 $value.CH2.wav  -c 1 $value.CH3.wav  -c 1 $value.CH4.wav  -c 1 $value.CH5.wav  -c 1 $value.CH6.wav ${value}_merged.wav
done
