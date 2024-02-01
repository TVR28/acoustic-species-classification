from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
import os
from PIL import Image
import torch
import ssl
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import splitfolders
import csv
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


def is_folder_empty(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return False

    # Check if the folder is empty
    for root, dirs, files in os.walk(folder_path):
        if not dirs and not files:
            print(f"The folder '{root}' is empty.")
            return True  # The folder is empty

    print(f"The folder '{folder_path}' is not empty.")
    return False

def get_subdirectories(folder_path):
    subdirectories = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            subdirectories.append(entry_path)
    return subdirectories

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    # Define the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the transform to resize and normalize the images
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
    ])

    # Update input folder and output folder paths
    input_folder = r"C:\Users\aahire\Documents\CSS581\Midterm\Data\IMAGES_HighPassFilter"
    output_folder = r"C:\Users\aahire\Documents\CSS581\Midterm\Data\Small_Dataset"

    ### Uncomment only for first time. once data is splitted into train and validation, comment it out
    #splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(0.8, 0.2), group_prefix=None)

    # Check for empty folders in the training and validation data
    train_path = os.path.join(output_folder, 'train')
    val_path = os.path.join(output_folder, 'val')

    train_directories = get_subdirectories(train_path)
    val_directories = get_subdirectories(val_path)

    for train_folder_path in train_directories:
        is_train_empty = is_folder_empty(train_folder_path)
    
    for val_folder_path in val_directories:
        is_val_empty = is_folder_empty(val_folder_path)
    
    if(is_train_empty or is_val_empty):
        print("Empty folder found for traiing and validation data, exiting")
        exit(0)

    # Create datasets for the training and testing sets
    train_dataset = torchvision.datasets.ImageFolder(output_folder + '/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(output_folder + '/val', transform=transform)
    train_size = len(train_dataset)
    val_size = len(val_dataset)

    # Create the data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True,num_workers=4)
    list_of_classes = os.listdir(r"C:\Users\aahire\Documents\CSS581\Midterm\Data\Small_Dataset\train")
    print(list_of_classes)
    classes = list(train_dataset.class_to_idx.keys())
    classes.sort()


    # Define the VGG model
    model = torchvision.models.vgg16(pretrained=True)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, len(list_of_classes))
    )

    ## uncomment for CPU
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    START_LR = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    model = model.to(device)
    criterion = criterion.to(device)

    ### uncomment for GPU
    model.cuda()
        # Train the model
    for epoch in range(20):
        ### uncomment for GPU
        torch.cuda.empty_cache()
        print('Epoch {}/{}'.format(epoch + 1, 20))
        print('-' * 20)

        running_loss = 0
        running_corrects = 0

        model.train()
        predictions = []
        true_labels = []
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.extend(labels.cpu().numpy())
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        report_dict = classification_report(true_labels, predictions, target_names=list_of_classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('training-classification-epoch' + str(epoch + 1) + '.csv')
        matrix = confusion_matrix(true_labels, predictions)
        cnf_matrix = confusion_matrix(true_labels, predictions)
        df_cm = pd.DataFrame(cnf_matrix / np.sum(cnf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('confusion-matrix-train-epoch' + str(epoch + 1) + '.csv')
        acc = matrix.diagonal()/matrix.sum(axis=1)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        ACC = (TP+TN)/(TP+FP+FN+TN)
        TPR = TP/(TP+FN)
        PPV = TP/(TP+FP)
        print("accuracy for all classes in train phase", ACC)
        print("recall for all classes in train phase", TPR)
        print("precision for all classes in train phase", PPV)
        
        pd.DataFrame(ACC, columns=['Accuracy']).to_csv('accuracy-train-epoch' + str(epoch + 1) + '.csv')
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        # Delete data to clear GPU memory
        del outputs
        del preds
        del labels
        del inputs
        torch.cuda.empty_cache()
        
        # Validation phase
        running_loss = 0
        running_corrects = 0
        model.eval()  # set the model to evaluation mode
        predictions = []
        true_labels = []
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.extend(labels.cpu().numpy())
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / val_size
        epoch_acc = running_corrects.double() / val_size
       
        #classification_report(true_labels, predictions, target_names=list_of_classes,output_dict=True)
        report_dict = classification_report(true_labels, predictions, target_names=list_of_classes,output_dict=True)
        report_pd = pd.DataFrame(report_dict)
        report_pd.to_csv('val-classification-epoch' + str(epoch + 1) + '.csv')
        matrix = confusion_matrix(true_labels, predictions)
        cnf_matrix = confusion_matrix(true_labels, predictions)
        df_cm = pd.DataFrame(cnf_matrix / np.sum(cnf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        df_cm.to_csv('confusion-matrix-val-epoch' + str(epoch + 1) + '.csv')
        acc = matrix.diagonal()/matrix.sum(axis=1)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        ACC = (TP+TN)/(TP+FP+FN+TN)
        TPR = TP/(TP+FN)
        PPV = TP/(TP+FP)
        pd.DataFrame(ACC, columns=['Accuracy']).to_csv('accuracy-val-epoch' + str(epoch + 1) + '.csv')
        print("accuracy for all classes in validation phase", ACC)
        print("recall for all classes in validation phase", TPR)
        print("precision for all classes in validation phase", PPV)
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        # Delete data to clear GPU memory
        del outputs
        del preds
        del labels
        del inputs
        torch.cuda.empty_cache()
                
        

    # Save the model
    torch.save(model, 'vgg16_model_gpu.pth')


