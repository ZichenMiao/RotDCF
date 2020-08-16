# ROTDCF: Decomposition Of Convolutional Filters For Rotation-Equivariant Deep Neural Netowrks
<font size=4>Code for paper: **ROTDCF: Decomposition Of Convolutional Filters For Rotation-Equivariant Deep Neural Netowrks**[[paper link]](https://openreview.net/pdf?id=H1gTEj09FX)

This repo implements the **rotMNIST** experiment in Section 4.1(ref. Table 3 and Table A.3).</font>

## Requirements
pytorch>=1.0
numpy>=1.18
matplotlib

## Dataset
Download processed rotMNIST dataset from [here](https://drive.google.com/file/d/1PsSvLh3wSux_oQ_7QlS3Q4yaQbSdBsxs/view?usp=sharing). Unzip it to this work directory.

Change *data_path* and *save_path* variables in dataset.py.


## Notebook for Rotation Equivariance
'Demo for Rotataion Equivariance' shows the main property of proposed method. Go through the notebook and you will observe rotation equivariance in output feature maps, e.g.,
<center class="half">
    <img src=./misc/featmap_rotequi_layer1.png width=600>
</center>

## Test Pretrained Models
### Test 3-layer CNN(M=32)
```python
./test.sh CNN 3
```
### Test 6-layer RotDCF(M=16, $N_theta=8$, K=5, $K_\alpha$=5)
change *M, $N_\theta$, K, and $K_\alpha$* in test.sh, and then,
```python
./test.sh RotDCF 6 
```

## Train New Models
### Train 3-layer CNN(M=32)
```python
./train.sh CNN 3
```
### Train 6-layer RotDCF(M=16, $N_theta=8$, K=5, $K_\alpha$=5)
change *M, $N_\theta$, K, and $K_\alpha$* in train.sh, and then,
```python
./train.sh RotDCF 6 
```