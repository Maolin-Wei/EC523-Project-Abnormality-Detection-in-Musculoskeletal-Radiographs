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
from torchcam.methods import SmoothGradCAMpp
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SAVE_PATH = './Models/Model_Architectures_Weighted_BCE_Adam/'
    batch_size = 128

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load testing data
    # _, _, test_data = read_data_from_csv()

    # test_dataset = MURADataset(test_data, transform=transform)

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 10)

    # models_list = [models.densenet169, models.resnet152, models.vit_b_16]
    models_list = [models.densenet169]

    for model_fn in models_list:
        model_name = model_fn.__name__
        model = model_fn(pretrained=False)
        # vision transformer
        if 'vit' in model_name:
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, 1)
        # densenet
        if 'densenet' in model_name:
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 1)
        # resnet
        if 'resnet' in model_name:
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)

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
        model.to(device)


        model.eval()
        img = read_image("sample_images/image1.png")
        if img.shape[0] == 4:
            img = img[:3]
        if img.shape[0] == 1:  # Convert grayscale to RGB
            img = img.repeat(3, 1, 1)
        print(img.shape)
        input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

        input_tensor = input_tensor.to(device)

        with SmoothGradCAMpp(model) as cam_extractor:
            out = model(input_tensor.unsqueeze(0))
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        # plt.imshow(activation_map[0].squeeze(0).cpu().numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

        # Resize the CAM and overlay it
        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        # Display
        plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

    print("Evaluation of all models completed.")
