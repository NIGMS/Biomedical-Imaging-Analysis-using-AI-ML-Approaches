
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms
import torchshow as ts
import medmnist
from medmnist import INFO, Evaluator




def Augment_Data(data_flag = 'breastmnist', download = True, batch_size = 16, data_transform = None,train_shuffle = False):
    
    '''
        Function for preparing and augmenting MedMNIST dataset for training. 
        Parameters
        ----------
        data_flag : the dataset to prepare.
        download : boolian if true then download dataset from the internet.
        batch_size : int that represents the portion of dataset to train at each round of training.
        data_transform : an object of torchvision transforms to be applied to data.
        train_shuffle : boolian that turn on and off randomly shuffling the data.
        
        Returns
        ----------
        plain_train_loader : object of torch dataloader used to batch plain training dataset into a set of portions of size batch_size.
        aug_train_loader : object of torch dataloader used to batch augmented training dataset into a set of portions of size batch_size.
        test_loader : object of torch dataloader used to batch testing dataset into a set of portions of size batch_size
        train_evaluator : object of MedMNIST Evaluator used to evaluate training dataset and return AUC and ACC.
        test_evaluator : object of MedMNIST Evaluator used to evaluate testing dataset and return AUC and ACC.
    '''
    
    # use dataclass provided by MedMNIST dataset library.
    DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
    
    basic_transform = transforms.Compose([transforms.ToTensor()])
    #, transforms.Normalize(mean=[.5], std=[.5]),
    plain_train_dataset = DataClass(split='train', transform=basic_transform, download=download)
    plain_train_loader = data.DataLoader(dataset=plain_train_dataset, batch_size=batch_size, shuffle = train_shuffle)
    # Load the data into training and testing datasets.
    
    if(data_transform == None):
        augmented_train_dataset = None
        aug_train_loader = None
        inputs, targets = next(iter(plain_train_loader))
        ts.show(inputs)
    else:
        augmented_train_dataset = DataClass(split='train', transform=data_transform, download=download)
        aug_train_loader = data.DataLoader(dataset=augmented_train_dataset, batch_size=batch_size, shuffle = train_shuffle)
        inputs, targets = next(iter(aug_train_loader))
        ts.show(inputs)
        
    test_dataset = DataClass(split='test', transform=basic_transform, download=download)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
   
    #set up evaluator for the experiment.
    train_evaluator = Evaluator(data_flag, 'train')
    test_evaluator = Evaluator(data_flag, 'test')
    
    return plain_train_loader,aug_train_loader,test_loader,train_evaluator,test_evaluator


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
    
    return task,n_classes
