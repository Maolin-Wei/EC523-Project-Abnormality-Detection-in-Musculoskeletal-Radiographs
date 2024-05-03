import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
from tqdm import tqdm
from process_data_from_csv import read_data_from_csv
from dataset import MURADataset
from csv_utils import *
from vit_models.modeling import VisionTransformer, CONFIGS
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image
from torchcam.utils import overlay_mask
import cv2



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SAVE_PATH = './Models/Model_Architectures_ViT/'
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load testing data
    # _, _, test_data = read_data_from_csv()

    # test_dataset = MURADataset(test_data, transform=transform)

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 10)

    config = CONFIGS['R50-ViT-B_16'] # "R50-ViT-B_16"
    num_classes = 1
    model_name = 'R50-ViT-B_16_4'
    model = VisionTransformer(config, 320, zero_head=True, num_classes=num_classes, vis=True)

    model_folder = os.path.join(SAVE_PATH, model_name)

    best_model_file = None
    for file in os.listdir(model_folder):
        if 'best' in file and file.endswith('.pth'):
            best_model_file = file
            break

    if best_model_file is None:
        raise FileNotFoundError(f"No best model found in {model_folder}")
    
    model_path = os.path.join(model_folder, best_model_file)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    im = Image.open("sample_images/image1.png")
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    if im.mode == 'L':
        im = im.convert('RGB')
    x = transform(im)
    x.size()

    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original Image')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)
    ax1.axis('off')
    ax2.axis('off')
    plt.show()
    probs = torch.nn.Softmax(dim=-1)(logits)
    top5 = torch.argsort(probs, dim=-1, descending=True)