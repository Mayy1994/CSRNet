# CSRNet


Cascaded Selective Resolution Network for Real-time Semantic Segmentation. 

The network architecture.
<p align="center"><img src="./figure/image1.png" width="1000" alt="" /></p>

Since real-time segmentation approaches tend to exploit lightweight networks, they mainly face three problems. 
- The poor performance in segmenting small structures like thin “poles”. 
- The “patch-like” predictions in the interior of the dominant objects. 
- Some classes with similar shapes and appearances, like car, bus and truck are confusing in classification. 
To alleviate these problems, we propose a multiple-stage segmentation network, CSRNet to refine the feature maps progressively. 


Segmentation comparison.
<p align="center"><img src="./figure/image2.png" width="800" alt="" /></p>



## Dependencies
- Python 3.6
- PyTorch 1.5.0
- CUDA 10.2 
- cuDNN 7.6.5

The experiments are conducted on a Ubuntu 18.04 LTS PC with two NVIDIA GeForce GTX 1080 Ti. Driver version is 460.91.03. GCC version is 7.5.0. Please refer to [`requirements.txt`](requirements.txt) for more details about the packages used.


## Installation
- Clone this repository.
```
git clone git@github.com:Mayy1994/CSRNet.git
```

## Dataset setup

Please download the [Cityscapes](https://www.cityscapes-dataset.com/downloads/) dataset.
- leftImg8bit_trainvaltest.zip (11GB)
- gtFine_trainvaltest.zip (241MB)


## Training
- Create a virtual environment and install the required python packages.
- Train CSRNet on Cityscapes dataset.
```
python train.py
```

## Evaluation
- Download the pretrained [CSRNet](https://drive.google.com/file/d/1JWNg4fXsde4or-Ke19_RPB2RYTPdQWVP/view?usp=sharing) and put it into $CSRNet_ROOT/model/ctyscapes/pretrained.
- Evaluate CSRNet on Cityscapes dataset.
```
python val.py
```

