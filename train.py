import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from process_data_from_csv import read_data_from_csv
from dataset import MURADataset
from csv_utils import *

torch.manual_seed(2024)

def train_and_validate_model_BCE(model, train_loader, valid_loader, criterion, optimizer, device, model_path, num_epochs=10):
    best_acc = 0.0
    model.train()

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.to(device)
            # Ensure labels are in the correct shape for BCEWithLogitsLoss
            labels = labels.to(device)
            labels = labels.view(-1, 1).float()  # Ensure labels are of the shape (batch_size, 1) if not already

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            # preds = torch.sigmoid(outputs).round()
            preds = (outputs > 0).float()
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        # Compute training statistics
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0

        for inputs, labels, _ in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            labels = labels.view(-1, 1)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            # preds = torch.sigmoid(outputs).round()
            preds = (outputs > 0).float()
            val_running_corrects += torch.sum(preds == labels.data)
            val_total += labels.size(0)

        # Compute validation statistics
        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_running_corrects.double() / val_total
        valid_loss_history.append(val_epoch_loss)
        valid_acc_history.append(val_epoch_acc.item())

        print(f'Epoch {epoch + 1}/{num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
              f'Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}')
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), f'{model_path}_best_model_epoch{epoch + 1}.pth')

    elapsed_time = time.time() - start_time

    return {
        'model': model,
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'valid_loss_history': valid_loss_history,
        'valid_acc_history': valid_acc_history,
        'elapsed_time': elapsed_time
    }


def train_and_validate_model_crossentropy(model, train_loader, valid_loader, criterion, optimizer, device, model_path, num_epochs=10):
    best_acc = 0.0

    # Set up the model training
    model.train()

    # Lists to keep track of progress
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0

        # Training phase
        for inputs, labels, _ in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0

        for inputs, labels, _ in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            val_total += labels.size(0)

        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_running_corrects.double() / val_total

        valid_loss_history.append(val_epoch_loss)
        valid_acc_history.append(val_epoch_acc.item())

        print(f'Epoch {epoch + 1}/{num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
              f'Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}')

        # Save the best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), f'{model_path}_best_model_epoch{epoch + 1}.pth')

    elapsed_time = time.time() - start_time

    return {
        'model': model,
        'train_loss_history': train_loss_history,
        'train_acc_history': train_acc_history,
        'valid_loss_history': valid_loss_history,
        'valid_acc_history': valid_acc_history,
        'elapsed_time': elapsed_time
    }

def calculate_pos_weight_from_dataloader(train_loader):
    num_positives = 0  # 1 is abnormal
    num_negatives = 0  # 0 is normal
    for _, labels, _ in train_loader:
        labels = labels.view(-1).numpy()
        num_positives += np.sum(labels)
        num_negatives += len(labels) - np.sum(labels)
    return torch.tensor([num_negatives / num_positives], dtype=torch.float)


def calculate_class_weights(train_loader):
    # Calculate the class weights inversely proportional to the class frequencies
    class_counts = torch.tensor([0.0, 0.0])
    for _, labels, _ in train_loader:
        labels = labels.view(-1)
        unique_labels, counts = labels.unique(return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_counts[label] += count
    total = class_counts.sum()
    class_weights = total / class_counts
    return class_weights

if __name__ == '__main__':
    # Define Hyperparameters
    # SAVE_PATH = '/content/drive/MyDrive/EC523_Project/Models/Model_Architectures'
    SAVE_PATH = './Models/Model_Architectures_Weighted_BCE_SGD'
    learning_rate = 0.0001 # 0.0001
    momentum = 0.9
    batch_size = 128
    num_epochs = 50
    crit = "BCELoss" #"CrossEntropy"
    opti = "SGD"
    transformations = "ReSize: 224X224, Rotation: 30deg, Vert_Flip: 50%, Hor_Flip: 50%, Normalization:(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define a transformation pipeline with normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Initialize the pretrained list of models
    # models_list = [models.densenet121, models.densenet161, models.densenet169, models.densenet201,
                #    models.resnext101_32x8d, models.resnext101_64x4d, models.resnet152, models.resnet50]

    # models_list = [models.densenet169, models.resnet152, models.vit_b_16, models.swin_b]

    models_list = [models.densenet169, models.resnet152]

    train_data, valid_data, test_data = read_data_from_csv()

    train_dataset = MURADataset(train_data, transform=train_transform)
    valid_dataset = MURADataset(valid_data, transform=valid_transform)
    test_dataset = MURADataset(test_data, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 10)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers = 10)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 10)

    pos_weight = calculate_pos_weight_from_dataloader(train_loader).to(device)
    print(f"Calculated pos_weight: {pos_weight}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # class_weights = calculate_class_weights(train_loader).to(device)
    # print(f"Calculated class_weights: {class_weights}")
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()

    for model_fn in models_list:
        model_name = model_fn.__name__
        model = model_fn(pretrained=True)
        # print(model)
        # vision transformer
        if 'swin' in model_name:
            num_features = model.head.in_features
            model.head = nn.Linear(num_features, 1)
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
            
        model = model.to(device)
        # Create paths for csv files
        model_folder = os.path.join(SAVE_PATH, model_name)
        model_path = os.path.join(model_folder, model_name)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        train_path = os.path.join(model_folder, 'train_history.csv')
        valid_path = os.path.join(model_folder, 'valid_history.csv')
        train_parameters_path = os.path.join(model_folder, 'train_parameters.csv')

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

        # model, train_loss_history, train_acc_history, elapsed_time = train_model(model, train_loader, criterion, optimizer, device, model_path, num_epochs=num_epochs)
        results = train_and_validate_model_BCE(model, train_loader, valid_loader, criterion, optimizer, device, model_path, num_epochs=num_epochs)
        model = results['model']
        train_loss_history = results['train_loss_history']
        train_acc_history = results['train_acc_history']
        valid_loss_history = results['valid_loss_history']
        valid_acc_history = results['valid_acc_history']
        elapsed_time = results['elapsed_time']
        # Save training results to CSV to their respective paths
        torch.save(model.state_dict(), f'{model_path}_final_model.pth')
        save_history_to_csv(train_loss_history, train_acc_history, train_path, is_training=True)
        save_history_to_csv(valid_loss_history, valid_acc_history, valid_path, is_training=False)
        save_model_training_parameters_to_csv(model_name,transformations,opti,learning_rate, momentum,batch_size,num_epochs,elapsed_time, train_parameters_path)
