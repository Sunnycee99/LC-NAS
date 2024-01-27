# Latency-Constrained Neural Architecture Search Method for Efficient Model Deployment on RISC-V Devices

This repository contains code and RISC-V latency dataset of the paper.

# Environment
## Hardware
- OS: Ubuntu 20.04
- CPU: Intel (R) Core (TM) i7-6950X CPU @ 3.00GHz
- GPU: Single NVIDIA TITAN X Pascal

## Software
- Pytorch 1.13.1
- Python 3.7.16
- nas-bench-201 1.3
- CUDA 12.0

# Setup
## Image Datasets
1. Download the [image datasets](https://drive.google.com/drive/folders/1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7).

2. Place the folders in ~path_to_lcnas/datasets/

## NAS-Bench-201
1. Download [NAS-Bench-201](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs).

2. Place the .pth file in ~path_to_lcnas/datasets/

# Running LC-NAS
```
./run.sh
```
This command uses latency predictor to search architecture within latency constraint 500 ms in different sample numbers on CIFAR-10. The resluts will be shown in the cmd.