# DG-LGK
This repository provides the official PyTorch implementation of the following paper:  
**Learning Generalized Knowledge from A Single Domain on Urban-scene Segmentation**

![avatar](https://github.com/leelxh/DG-LGK/blob/main/figure/LGK.png)
*Semantic segmentation results in unseen real-world cases by our model trained on the synthetic dataset GTA5. The samples are from the BDD-100k, Mapillary, ACDC-night, ACDC-fog, ACDC-snow, and ACDC-rain datasets, respectively.*

## Congifuration Environment
- python 3.7
- pytorch 1.8.2
- torchvision 0.9.2
- cuda 11.1

## Before start
Download the GTA5, SYNTHIA, Cityscapes, BDD-100K, Mapillary, and ACDC datasets.

## Training and test
Run like this: python main.py --gta5_data_path /dataset/GTA5 --city_data_path /dataset/cityscapes --cuda_device_id 0
