# Cross-modal Guided Visual Representation Learning for Social Image Retrieval

[toc]

## 1. Introduction


This repository provides the code for our paper "Cross-modal Guided Visual Representation Learning for Social Image Retrieval"

Cross-modal Guided Visual Representation Learning for Social Image Retrieval


In the following, we will guide you how to use this repository step by step. 

## 2. Preparation

```bash
unzip CGVR-code.zip
cd CGVR-code
```

### 2.1 Requirements

- python 3.7.10
- gensim 3.8.0
- nltk 3.6.2
- numpy 1.20.2
- Pillow 9.3.0
- torch 1.9.0
- torchvision 0.10.0
- tqdm 4.62.2


### 2.2 Download image datasets.

Before running the code, we need to make sure that everything needed is ready. 

- The `data/` folder is the collection of data splits for MirFlickr25K and NUS-WIDE datasets. The raw images of Flickr25K and NUS-WIDE datasets should be downloaded additionally and arranged in `data/mirflickr25k` and `data/nuswide` respectively. 
  

## 3. Train and Distillation

If you want to train and test CGVR-64bits model on NUS-WIDE or MirFlickr25k dataset, you can do
```bash
python train-test.py --dataname 'nuswide' --CGVR_nbits 64
or
python train-test.py --dataname 'mirflickr25k' --CGVR_nbits 64
```
After running, log and model files will be saved under`checkpoints` of the corresponding dataset folder.
