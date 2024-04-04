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
import time
from process_data_from_csv import read_data_from_csv
from dataset import MURADataset
import csv

torch.manual_seed(42)

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

def plot_roc_curve(fpr, tpr, roc_auc, model_path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_path}_roc_curve.png')
    plt.close()  # Close the plot to avoid overlapping plots
    # plt.show()

#Define Hyperparameters
# SAVE_PATH = '/content/drive/MyDrive/EC523_Project/Models/Model_Architectures'
SAVE_PATH = './Models/Model_Architectures'
learning_rate = 0.0001
momentum = 0.9
batch_size = 8
num_epochs = 100
crit = "CrossEntropy"
opti = "SGD"
transformations = "ReSize: 320X320, Rotation: 30deg, Vert_Flip: 50%, Hor_Flip: 50%, Normalization:(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a transformation pipeline with normalization
train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the pretrained list of models
models_list = [models.densenet121, models.densenet161, models.densenet169, models.densenet201,
               models.resnext101_32x8d, models.resnext101_64x4d, models.resnet152, models.resnet50]

criterion = nn.CrossEntropyLoss()

train_data, valid_data = read_data_from_csv()

train_dataset = MURADataset(train_data, transform=train_transform)
valid_dataset = MURADataset(valid_data, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 10)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers = 10)

def save_history_to_csv(model_name, train_loss_history, train_acc_history, path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy'])
        for epoch, (loss, acc) in enumerate(zip(train_loss_history, train_acc_history), 1):
            writer.writerow([epoch, loss, acc])

def save_model_training_parameters_to_csv(model_name,transformations,optimizer,momentum,batch_size,num_epochs, elapsed_time, path):
  with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Name', 'Transformations', 'Optimizer', 'Learning Rate', 'Momentum', 'Batch Size', 'Number of Epochs', 'Training Time'])
        writer.writerow([model_name, transformations, optimizer, learning_rate, momentum, batch_size, num_epochs, elapsed_time])

def save_validation_to_csv(metrics_data, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics_data)

def train_model(model, train_loader, criterion, optimizer, device, model_path, num_epochs=10):
    model.train()

    # Lists to keep track of progress
    epoch_loss_history = []
    epoch_acc_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        epoch_loss_history.append(epoch_loss)
        epoch_acc_history.append(epoch_acc.item())

        print(f'Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'{model_path}_epoch_{epoch}_model.pth')

    elapsed_time = time.time() - start_time
    return model, epoch_loss_history, epoch_acc_history, elapsed_time


def validate_model(model, valid_loader, device, model_path):
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
    plot_roc_curve(fpr, tpr, roc_auc, model_path)
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

for model_fn in models_list:
    model_name = model_fn.__name__
    model = model_fn(pretrained=True)
    model = model.to(device)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 2)

    # Create paths for csv files
    model_folder = os.path.join(SAVE_PATH, model_name)
    model_path = os.path.join(model_folder, model_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_path = os.path.join(model_folder, 'train_history.csv')
    train_parameters_path = os.path.join(model_folder, 'train_parameters.csv')
    model_results_path = os.path.join(model_folder, 'validation_results.csv')

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    model, train_loss_history, train_acc_history,elapsed_time = train_model(model, train_loader, criterion, optimizer, device, model_path, num_epochs=num_epochs)
    metrics_data = validate_model(model, valid_loader, device, model_path)

    # Save training results to CSV to their respective paths
    torch.save(model.state_dict(), f'{model_path}_final_model.pth')
    save_history_to_csv(model_name, train_loss_history, train_acc_history,train_path)
    save_model_training_parameters_to_csv(model_name,transformations,opti,momentum,batch_size,num_epochs,elapsed_time, train_parameters_path)
    save_validation_to_csv(metrics_data, model_results_path)