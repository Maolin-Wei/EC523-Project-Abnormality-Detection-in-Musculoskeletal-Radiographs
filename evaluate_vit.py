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


def plot_training_history(epoch_loss_history, epoch_acc_history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_loss_history, label='Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(len(epoch_loss_history)))
    plt.legend()

    plt.subplot(1, 2, 2)
    print('epoch_acc_history:', epoch_acc_history)
    plt.plot(epoch_acc_history, label='Accuracy')

    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(epoch_acc_history)))
    plt.legend()

    plt.show()
    
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
    plt.close() 
    # plt.show()

def test_model(model, valid_loader, device, model_folder, model_name):
    model.eval()  # Set model to evaluate mode

    all_preds = []
    all_labels = []
    parts_list = []  # To store body part information for each prediction

    with torch.no_grad():
        for inputs, labels, parts in tqdm(valid_loader): 
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # _, preds = torch.max(outputs, 1)  # crossentropy loss
            # print(outputs)
            # preds = torch.sigmoid(outputs).round()  # BCE loss
            preds = (outputs > 0).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            parts_list.extend(parts)

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
            'sensitivity': round(sensitivity, 3),
            'specificity': round(specificity, 3),
            'accuracy': round(accuracy, 3)
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

    metrics_data.append(['Overall', round(overall_sensitivity, 3), round(overall_specificity, 3), round(overall_accuracy, 3)])

    metrics_data.append(['fpr', 'tpr', 'thresholds', 'roc_auc'])

    for i in range(len(fpr)):
        metrics_data.append([fpr[i], tpr[i], thresholds[i], roc_auc])

    return metrics_data


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
    _, _, test_data = read_data_from_csv()

    test_dataset = MURADataset(test_data, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 10)

    config = CONFIGS['R50-ViT-B_16'] # "R50-ViT-B_16"
    num_classes = 1
    model_name = 'R50-ViT-B_16_4'
    model = VisionTransformer(config, 320, zero_head=True, num_classes=num_classes)

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

    metrics_data = test_model(model, test_loader, device, model_folder, model_name)
    print(f'Evaluation for {model_name}:')
    print(metrics_data)
    model_results_path = os.path.join(model_folder, 'test_results.csv')
    save_validation_to_csv(metrics_data, model_results_path)

    print("Evaluation of all models completed.")
