# Towards OOD detection in fine-grained environments

This repository contains the code necessary to replicate the results of our paper:

***Mixture Outlier Exposure: Towards Out-of-Distribution Detection in Fine-grained Environments***

[Jingyang Zhang](https://zjysteven.github.io/), Nathan Inkawhich, Randolph Linderman, Yiran Chen, Hai Li @ [Duke CEI Lab](https://cei.pratt.duke.edu/)

Paper (arxiv preprint): https://arxiv.org/abs/2106.03917

This work has been accepted to WACV'23!!

## Overview

### Motivation

<p align="center">
    <img src='/figures/overview.png' width='870'>
</p>
<p>
    <em>Left: A comparison of OOD detection in coarse- and fine-grained environments. Intuitively, fine-grained detection is much more challenging. Right: A *conceptual* illustration of MixOE. A Standard model with no OOD considerations tends to be over-confident on OOD samples. OE is able to calibrate the prediction confidence on coarse-grained OOD, but the outputs on fine-grained OOD are uncontrolled (marked by ``?''). MixOE aims for a smooth decay of the confidence as the inputs transition from ID to OOD, and thus enables detection of both coarse/fine-grained OOD.</em>
</p>


The capability of detecting Out-of-distribution (OOD) samples that do not belong to one of the known classes of DNNs during inference time is crucial for reliable operations in the wild.
Existing works typically use *coarse-grained* benchmarks (e.g., CIFAR-10 v.s. SVHN/LSUN) to perform evaluation, which fail to approximate many real-world scenarios which inherently have *fine-grained* attributes (e.g., bird species recognition, medical image classification).
In such fine-grained environments, one may expect OOD samples to be highly granular w.r.t. in-distribution (ID) data, which intuitively can be very difficult to identify.
**Unfortunately, OOD detection in fine-grained environments remains largely underexplored.**

### Our contributions
1. **We construct four large-scale, fine-grained test environments for OOD detection.** The test benches are generated using a holdout-class method on public fine-grained classification datasets. Later we have detailed instructions for you to prepare the datasets and easily reproduce our test environments.

<p align="center">
    <img src='/figures/initial_eval.png' width='650'>
</p>
<p>
    <em>TNR95 of existing methods against coarse-grained (first row) and fine-grained OOD data (second row). The gray dashed line is the baseline performance (MSP). Fine-grained novelties are significantly harder to detect in all datasets for all methods. Also note how the methods that utilize outlier data (coral ones) help with coarse-grained OOD but barely improve fine-grained detection rates.</em>
</p>

2. Through initial evaluation, **we find that existing methods struggle to detect fine-grained novelties** (underperforming a simple baseline [MSP](https://arxiv.org/abs/1610.02136)). Even the methods that explicitly incorporate auxiliary outlier data (e.g., [Outlier Exposure](https://arxiv.org/abs/1812.04606)) does *not* help much. We further conduct analysis to explain why this is the case.

<p align="center">
    <img src='/figures/method_vis.png' width='870'>
</p>
<p>
    <em>Visualization of the data samples (second row) and their representations in the DNN's feature space (first row). The color lightness in (d)/(e) indicates the prediction confidence encoded in the soft target of each corresponding outlier sample. Note, (b) and (c) are test OOD samples and are never seen during the training. The empirical training outliers (d) enclose the region where coarse OOD data (b) locate but does not cover the much broader area the fine OOD samples (c) span. MixOE mixes the ID (a) and training outliers (d) to induce larger coverage which accounts for both coarse- and fine-grained novelties. Moreover, the soft targets of the mixed data will calibrate the model's prediction confidence to smoothly decay from ID to OOD.</em>
</p>

3. **We propose Mixture Outlier Exposure (MixOE)**, a novel method for OOD detection in fine-grained environments. MixOE consistently improves the detection rates against both fine- and coarse-grained OOD samples across all the four test benches.

See our paper for details!

## Get started
### Environment
Follow the commands below to set up the environment.

1. Clone the repo: `git clone https://github.com/zjysteven/MixOE.git`

2. Create a conda environment
```
conda create -n mixoe python=3.8
conda activate mixoe
python -m pip install -r requirements.txt
```

### Dataset
Please see detailed instructions [here](/data/README.md).

## Reproducing experiments
We provide sample scripts `train.sh` and `eval.sh` in `scripts/` folder showcasing how to train/evaluate our method and other considered methods. Note that all OE-based methods (including MixOE) fine-tune the trained baseline model, so to run OE/MixOE you first need to do standard training with `train_baseline.py`. For everyone's convenience, we have uploaded the trained baseline weights [here](https://drive.google.com/drive/folders/1XdGWCgYR1lBiJZIuhlO0e3VHisJ8MDlH?usp=sharing).

## Reference
If you find our work/code helpful, please consider citing our work.
```
Coming soon!
```
