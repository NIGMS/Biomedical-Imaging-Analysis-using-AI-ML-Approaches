
from tqdm import tqdm
#import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms
#from IPython.display import Image
#import matplotlib.pyplot as plt
import medmnist
from torchshow import show

from medmnist import INFO, Evaluator

def Prepare_Data(data_flag = 'pathmnist', download = True, batch_size = 128):
    
    '''
        Function for preparing MedMNIST dataset for training. 
        Parameters
        ----------
        data_flag : the dataset to prepare.
        download : boolian if true then download dataset from the internet.
        batch_size : int that represents the portion of dataset to train at each round of training.
        
        Returns
        ----------
        train_loader : object of torch dataloader used to batch training dataset into a set of portions of size batch_size.
        train_loader_at_eval : object of torch dataloader used to batch evaluation dataset into a set of portions of size batch_size
        test_loader : object of torch dataloader used to batch testing dataset into a set of portions of size batch_size
        train_evaluator : object of MedMNIST Evaluator used to evaluate training dataset and return AUC and ACC.
        test_evaluator : object of MedMNIST Evaluator used to evaluate testing dataset and return AUC and ACC.
    '''
        
    # use dataclass provided by MedMNIST dataset library.
    DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
    
    # Data augmentation and normalization for training.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Load the data into training and testing datasets.
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    
       
    # Encapsulate data into dataloader form (split the data into batches for processing).
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    #show sample of training set
    inputs, targets = next(iter(train_loader))
    show(inputs[:20])
    
    #set up evaluator for the experiment.
    train_evaluator = Evaluator(data_flag, 'val')
    test_evaluator = Evaluator(data_flag, 'test')
    return (train_loader, train_loader_at_eval, test_loader, train_evaluator, test_evaluator)


def Get_DataSet_Information(data_flag = 'pathmnist'):
    
    '''
        Function for getting information about a specific dataset from the MedMNIST datasets. 
        Parameters
        ----------
        data_flag : the dataset to prepare.
                
        Returns
        ----------
        task : string represent the task of the dataset specified by data_flag.
        n_classes : int represents the number of classes of the dataset specified by data_flag.
    '''
    
    task = INFO[data_flag]['task']
    # Number of classes in the dataset
    n_classes = len(INFO[data_flag]['label'])
    return (task,n_classes)
