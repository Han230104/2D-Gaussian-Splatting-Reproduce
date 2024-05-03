# 2D-Gaussian-Splatting-Reproduce
This repository contains the unofficial implementation of the paper ["2D Gaussian Splatting for Geometrically Accurate Radiance Fields"](https://arxiv.org/pdf/2403.17888).
# Installation
### Clone the repository 
```
git clone git@github.com:Han230104/2D-Gaussian-Splatting-Reproduce.git
```
or
```
# HTTPS
git clone https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce.git
```
### Create an anaconda environment
```
cd 2d-gaussian-splatting-reproduce
conda env create --file environment.yml
conda activate 2dgs
```
### Download dataset 
Create a folder to store the dataset
```
mkdir datasets
```
The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/).
```
- datasets
  - bicycle
  - bonsai
  - counter
  - flowers
  - garden
  - kitchen
  - playroom
  - stump
  - treehill
```
### Training and Evaluation
```
# Mip-NeRF 360 dataset
python run_mipnerf360.py
```
### Results
You will get similar results like this on Mip-NeRF 360 dataset:
##### Ourdoor
| Model    | PSNR ↑     | SSIM ↑    | LIPPS ↓|  
| ------   | ------     | ------    | ------ |  
| 2DGS     | 24.33      |  0.709    | 0.284  |  
| Ours     | 24.25      |  0.711    | 0.278  |   
##### Indoor
| Model    | PSNR ↑     | SSIM ↑    | LIPPS ↓|  
| ------   | ------     | ------    | ------ |  
| 2DGS     | 30.39      |  0.924    | 0.182  |  
| Ours     | 30.28      |  0.923    | 0.181  |   
# Acknowledgements
This project is built upon [2DGS](https://surfsplatting.github.io/) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). We also taken some code from [gaussian-opacity-fields](https://github.com/autonomousvision/gaussian-opacity-fields). We thank all the authors for their great work and repos. 
