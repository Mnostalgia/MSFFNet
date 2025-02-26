# MSFFNet
Semantic Segmentation of Agricultural Crop Multispectral Image Using Feature Fusion



This repository contains the code for my paper: [Semantic Segmentation of Agricultural Crop Multispectral Image Using Feature Fusion
](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003077667).



<p float="center">
  <img src="./readme/model.png" width="900" />
</p>




## Custom dataset
Due to circumstances, we are unable to provide our own dataset. sorry.

### DATA-DIR
```sh
<dataset>
│
├── rgb               # rgb images
│   ├── image_1.JPG
│   ├── image_2.JPG
│   └── ...
│
├── etc multispectral  # another multispectral images dic ex) blue, green, nir, re ---
│   ├── image_1.TIF
│   ├── image_2.TIF
│   └── ...
│
├── labels             # label images
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
│
├── train.txt          
└── test.txt           # Train-test split.

<train.txt>
image_1
image_2
...

<test.txt>
...
image_n-1
image_n

```
We used ndvi as the name of the etc multispectral folder and used the TIF image as the dataset.

In the __getitem__ function of class MF_dataset in MF_dataset.py, change the folder name and extension name to suit your dataset.

## Getting started

This code was tested on `linux` and requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)


### 1. Setup environment

```shell
conda create --name <your env name> --file environment.yml
```



### 2. Run

## Train


```shell
python train.py
```
The results will be rendered and put in ./runs/


## Inference

```shell
python inference.py
```
The results will be rendered and put in ./result/


### Inference example
<p float="left">
  <img src="./readme/inference.png" width="300" />
</p>
