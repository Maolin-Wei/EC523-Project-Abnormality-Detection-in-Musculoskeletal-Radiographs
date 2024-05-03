import csv

def save_history_to_csv(train_loss_history, train_acc_history, path, is_training):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if is_training:
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy'])
        else:
            writer.writerow(['Epoch', 'Valid Loss', 'Valid Accuracy'])
        for epoch, (loss, acc) in enumerate(zip(train_loss_history, train_acc_history), 1):
            writer.writerow([epoch, loss, acc])

def save_model_training_parameters_to_csv(model_name,transformations,optimizer, learning_rate, momentum,batch_size, num_epochs, elapsed_time, path):
  with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Name', 'Transformations', 'Optimizer', 'Learning Rate', 'Momentum', 'Batch Size', 'Number of Epochs', 'Training Time'])
        writer.writerow([model_name, transformations, optimizer, learning_rate, momentum, batch_size, num_epochs, elapsed_time])

def save_validation_to_csv(metrics_data, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics_data)