# 2D-Gaussian-Splatting-Reproduce
This repository contains the unofficial implementation of the paper ["2D Gaussian Splatting for Geometrically Accurate Radiance Fields"](https://arxiv.org/pdf/2403.17888).

GVL lab, University of Southern California
## Overview
<p float="left">
  <img src="https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce/blob/master/assets/garden-rgb.png?raw=true" width="350" />
   <img src="https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce/blob/master/assets/kitchen-rgb.png?raw=true" width="350" />
</p>

**Rendered** RGB Image: garden(left), kitchen(right)

## Installation
### Clone the repository 
```
# SSH
git clone git@github.com:Han230104/2D-Gaussian-Splatting-Reproduce.git
```
or
```
# HTTPS
git clone https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce.git
```
### Create an anaconda environment
```
cd 2D-Gaussian-Splatting-Reproduce
conda env create --file environment.yml
conda activate 2dgs

pip install -r requirements.txt
```
## Download dataset 
Create a folder to store the dataset
```
mkdir datasets
```
#### MpiNeRF360
The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/).

After downloading the dataset, you should organize your data like this:
```
- 2D-Gaussian-Splatting-Reproduce
  - datasets
    - bicycle
    - bonsai
    - counter
    - flowers
    - garden
    - kitchen
    - room
    - stump
    - treehill
```
#### DTU
You can download the preprocessed data from [here](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9).

You also need to download the ground truth [DTU point cloud](https://roboimagedata.compute.dtu.dk/?page_id=36).

After downloading the dataset, you should organize your data like this:
```
- 2D-Gaussian-Splatting-Reproduce
  - datasets
    - DTU_mask  # preprocessed data
      - scan105
      ...
    - DTU  # official data
      - Points
      - ObsMask
      - Calibration
```

## Training and Evaluation
Run the training and evaluation script
```
# Mip-NeRF 360 dataset
python run_mipnerf360.py
```

```
# DTU dataset
python run_dtu.py
```

## Results
You will get similar results like this on Mip-NeRF 360 dataset:
#### Ourdoor
| Model    | PSNR ↑     | SSIM ↑    | LIPPS ↓|  
| ------   | ------     | ------    | ------ |  
| 2DGS     | **24.33**      |  0.709    | 0.284  |  
| Ours     | 24.25      |  **0.711**    | **0.278**  |   
#### Indoor
| Model    | PSNR ↑     | SSIM ↑    | LIPPS ↓|  
| ------   | ------     | ------    | ------ |  
| 2DGS     | 30.39      |  0.924    | 0.182  |  
| Ours     | **30.53**      |  **0.925**    | **0.178**  |   
# Acknowledgements
This project is built upon [2DGS](https://surfsplatting.github.io/) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). We also borrow some code from [gaussian-opacity-fields](https://github.com/autonomousvision/gaussian-opacity-fields). We thank all the authors for their great work and repos. 
