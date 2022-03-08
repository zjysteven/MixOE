# Exploiting Adversarial Training models to compromise privacy

This repository contains the code necessary to replicate the results of our paper:

***Mixture Outlier Exposure: Towards Out-of-Distribution Detection in Fine-grained Environments***

[Jingyang Zhang](https://zjysteven.github.io/), Nathan Inkawhich, Randolph Linderman, Yiran Chen, Hai Li @ [Duke CEI Lab](https://cei.pratt.duke.edu/)

Paper (arxiv preprint): Coming soon!


## Overview

### Motivation

![overview](/figures/overview.png)

<p align="center">
    <img src='/figures/overview.png'>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
    Left: A comparison of OOD detection in coarse- and fine-grained environments. Intuitively, fine-grained detection is much more challenging. Intuitively, fine-grained settings are more challenging. Right: A *conceptual* illustration of MixOE. A Standard model with no OOD considerations tends to be over-confident on OOD samples. OE is able to calibrate the prediction confidence on coarse-grained OOD, but the outputs on fine-grained OOD are uncontrolled (marked by ``?''). MixOE aims for a smooth decay of the confidence as the inputs transition from ID to OOD, and thus enables detection of both coarse/fine-grained OOD.
    </div>
</p>

The capability of detecting Out-of-distribution (OOD) samples that do not belong to one of the known classes of DNNs during inference time is crucial for reliable operations in the wild.
Existing works typically use *coarse-grained* benchmarks (e.g., CIFAR-10 v.s. SVHN/LSUN) to perform evaluation, which fail to approximate many real-world scenarios which inherently have *fine-grained* attributes (e.g., bird species recognition, medical image classification).
In such fine-grained environments, one may expect OOD samples to be highly granular w.r.t. in-distribution (ID) data, which intuitively can be very difficult to identify.
**Unfortunately, OOD detection in fine-grained environments remains largely underexplored.**

### Our contributions



## Get started
### Environment
Follow the commands below to set up the environment.

1. Clone the repo: `git clone https://github.com/zjysteven/PrivayAttack_AT_FL.git`

2. Create a conda environment
```
conda create -n AT-privacy python=3.8
conda activate AT-privacy
python -m pip install -r requirements.txt
```

### Dataset
We use [ImageNet](https://www.image-net.org/) as the dataset. To run our experiments, make sure that you download the ImageNet and have `train/`, `val/` subfolder inside the root directory. 

### Models
We use pre-trained models from [robust-transfer](https://github.com/microsoft/robust-models-transfer) repo. `download_pretrained_models.sh` is a sample download script. 

## Reproducing experiments
All experiments can be reproduced by running `main.py`. We provide a sample script `main.sh`. Remember to change the directory of ImageNet to your own version.

## Reference
If you find our work/code helpful, please consider citing our work.
```
@article{zhang2022privacy,
  title={Privacy Leakage of Adversarial Training Models in Federated Learning Systems},
  author={Zhang, Jingyang and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2202.10546},
  year={2022}
}
```