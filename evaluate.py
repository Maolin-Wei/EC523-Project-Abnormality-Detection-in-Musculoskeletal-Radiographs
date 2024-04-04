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
import csv

def save_validation_to_csv(metrics_data, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics_data)


def plot_roc_curve(fpr, tpr, roc_auc, model_folder, model_name):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_folder, f'{model_name}_roc_curve.png'))
    plt.close()  # Close the plot to avoid overlapping plots
    # plt.show()

def validate_model(model, valid_loader, device, model_folder, model_name):
    model.eval()  # Set model to evaluate mode

    all_preds = []
    all_labels = []
    parts_list = []  # To store body part information for each prediction

    with torch.no_grad():
        for inputs, labels, parts in tqdm(valid_loader):  # Assuming body part info is available
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            parts_list.extend(parts)  # Assuming 'parts' is a list or similar iterable
    #Calculate Values for Sensitivity and Specificity graph
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, model_folder, model_name)
    # Calculate metrics for each body part
    parts_set = set(parts_list)  # Get unique body parts
    body_part_metrics = {}

    for part in parts_set:
        part_indices = [i for i, p in enumerate(parts_list) if p == part]  # Indices of this part
        part_preds = np.array([all_preds[i] for i in part_indices])
        part_labels = np.array([all_labels[i] for i in part_indices])

        tn, fp, fn, tp = confusion_matrix(part_labels, part_preds).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        body_part_metrics[part] = {
            'sensitivity': round(sensitivity, 2),
            'specificity': round(specificity, 2),
            'accuracy': round(accuracy, 2)
        }

    # Calculate overall metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    overall_sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    overall_specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    overall_accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Prepare data to save
    metrics_data = [
        ['Body Part', 'Sensitivity', 'Specificity', 'Accuracy']
    ]

    for part, metrics in body_part_metrics.items():
        metrics_data.append([part, metrics['sensitivity'], metrics['specificity'], metrics['accuracy']])

    metrics_data.append(['Overall', round(overall_sensitivity, 2), round(overall_specificity, 2), round(overall_accuracy, 2)])

    metrics_data.append(['fpr', 'tpr', 'thresholds', 'roc_auc'])

    for i in range(len(fpr)):
        metrics_data.append([fpr[i], tpr[i], thresholds[i], roc_auc])

    return metrics_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PATH = './Models/Model_Architectures'
batch_size = 8

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load testing data
_, test_data = read_data_from_csv()

test_dataset = MURADataset(test_data, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 10)

models_list = [models.densenet121, models.densenet161, models.densenet169, models.densenet201,
               models.resnext101_32x8d, models.resnext101_64x4d, models.resnet152, models.resnet50]

for model_fn in models_list:
    model_name = model_fn.__name__
    model = model_fn(pretrained=False) 
    model_folder = os.path.join(SAVE_PATH, model_name)
    model_path = os.path.join(model_folder, f'{model_name}_final_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    metrics_data = validate_model(model, test_loader, device, model_folder, model_name)
    print(f'Evaluation for {model_name}:')
    print(metrics_data)
    model_results_path = os.path.join(model_folder, 'validation_results.csv')
    save_validation_to_csv(metrics_data, model_results_path)

print("Evaluation of all models completed.")
