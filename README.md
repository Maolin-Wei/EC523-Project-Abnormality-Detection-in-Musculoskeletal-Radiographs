# EC523-Project-Abnormality-Detection-in-Musculoskeletal-Radiographs
## Team Members
Manuel Segimón Plana, Maolin Wei, Daniel Vinals-Garcia, Manasvini.V.Srinivasan

## Introduction
Our task is to develop an automated abnormality detection system for musculoskeletal radiographs, similar to the system described in the referenced paper. The system will analyze radiographs to identify fractures, lesions, or joint abnormalities. One of the main challenges of this task is accurately identifying subtle abnormalities within complex anatomical structures while accounting for variations in imaging conditions and patient anatomy.

![Task](https://github.com/Maolin-Wei/EC523-Project-Abnormality-Detection-in-Musculoskeletal-Radiographs/assets/144057115/e8cc2835-149a-4aac-ae67-93296b2317bd)

## Installation
1. Create a virtual environment `conda create -n [env_name] python=3.10` and activate it `conda activate [env_name]`
2. Install [Pytorch](https://pytorch.org/get-started/locally/)
3. Install other packages by the command `pip install matplotlib numpy tqdm pandas scikit-learn Pillow`

## Dataset
The dataset we use is [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/), which contains 40,561 images from 14,863 musculoskeletal studies.

Download it and put in the root folder, e.g. ./MURA-v1.1/

## Training

The dataset is divided into training, testing, and validation sets following an 80/10/10 split.

**Train with the ResNet, DenseNet, etc. models**
```bash
python train.py
```

**Train with Vision Transformer**
```bash
python train_vit.py
```

The `hyperparameters` and `data_path` can be adjust in the code

## Evaluation
**Evaluate with the ResNet, DenseNet, etc. models**
```bash
python evaluate.py
```

**Evaluate with Vision Transformer**
```bash
python evaluate_vit.py
```

## Generate Class Activation Maps for ResNet and DenseNet
```bash
python get_CAM.py
```

## Generate Attention Map for Vision Transformer
```bash
python visualize_attention_map.py
```

## Acknowledgements
- We appreciate the Stanford ML Group for their proposed [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/)
- We appreciate the code for [ViT implementation](https://github.com/lucidrains/vit-pytorch)
