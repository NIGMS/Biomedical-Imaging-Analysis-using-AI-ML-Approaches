
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms

from tqdm import tqdm

def test_loop(model,data_loader,evaluator):
    
    '''
        Function for running testing loop
        Parameters
        ----------
        model : object of ResNet-18 model initialized from Model.py
        data_loader : object returned from Prepare_data in PrepareDataset.py 
        evaluator : object from medmnist
        
        Returns
        ----------
        metrics : AUC and ACC from testing dataset evaluation.
    ''' 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    ## set GPU device
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
   
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs = inputs.to(device)   ## load to GPU
            targets = targets.to(device)   ## load to GPU
            outputs = model(inputs)
            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().detach().numpy()
        metrics = evaluator.evaluate(y_score)
        print('%s  auc: %.3f  acc:%.3f' % (evaluator.split, *metrics))
        
        return metrics

def train_loop(model, train_loader, train_loader_eval, criterion, optimizer,evaluator, num_epochs=25):
    
    '''
        Function for running training loop
        Parameters
        ----------
        model : object of ResNet-18 model initialized from Model.py
        train_loader : object returned from Prepare_data in PrepareDataset.py represents train dataset.
        train_loader_eval : object returned from Prepare_data in PrepareDataset.py represents evaluation dataset.
        criterion : object from Model.py represent model loss function criterion.
        optimizer : object from Model.py represent model optimizer.
        evaluator : object from medmnist.
        num_epochs : int represents the number of training epochs.
        
        Returns
        ----------
        model : object of ResNet-18 model trained on training dataset
    ''' 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    ## set GPU device
    val_acc_history = []
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            inputs = inputs.to(device)    ## load to GPU
            targets = targets.to(device)  ## load to GPU
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if(train_loader_eval != None):
            print('==> Evaluating at Epoch ',epoch, '...')
            metrics = test_loop(model,train_loader_eval,evaluator)
            val_acc_history.append(format(metrics[1],".23"))
            
    return model

def aug_train_loop(model, plain_train_loader, aug_train_loader, criterion, optimizer,evaluator, num_epochs=25):
    
    '''
        Function for running augmented dataset training loop and training input model.
        Parameters
        ----------
        model : object of ResNet-18 model initialized from Model.py
        plain_train_loader : object returned from Augment_Data in PrepareDataset.py represents plain train dataset.
        aug_train_loader : object returned from Augment_Data in PrepareDataset.py represents augmented train dataset.
        criterion : object from Model.py represent model loss function criterion.
        optimizer : object from Model.py represent model optimizer.
        evaluator : object from medmnist.
        num_epochs : int represents the number of training epochs.
        
    ''' 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    ## set GPU device
    val_acc_history = []
    for epoch in range(num_epochs):
        batch_size = plain_train_loader.batch_size
        num_batches = int(len(plain_train_loader.dataset)/batch_size)
        #print('batch_size',batch_size)
        #print('num_batches',num_batches)
        model.train()
        for i in tqdm(range(num_batches*2)):
            if(i%2 == 0):
                inputs, targets =  next(iter(plain_train_loader))
            else:
                inputs, targets =  next(iter(aug_train_loader))
            # forward + backward + optimize
            optimizer.zero_grad()
            inputs = inputs.to(device)    ## load to GPU
            targets = targets.to(device)  ## load to GPU
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model


