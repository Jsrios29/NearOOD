"""
Created on Tue Oct 12 20:56:35 2021

Description: This File defines methods that help train the model

@author: Juan Rios
"""

#%% ----- Imports -------------------------------------------------------------

import time
import copy
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import torch.nn.functional as F

#%% -----Helper Methods -------------------------------------------------------

"""
This methods takes a model and trains and validates on the validation set for each epoch.
Each epoch checks if the current model is the best model in terms of its validation accuracy.
If so, the model is saved and returned.

model - the architecture (e.g. densenet)
dataloaders - 
criterion - the loss function
optimizer - the alogirthm to optimize (e.g. SGD)
num_epochs - how many epochs to train for
is_inception - indicates this model is not inception
"""
def train_model(model, dataloaders, criterion, optimizer, num_epochs,
                scheduler, device):

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Detect if we have a GPU available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                   
                   
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Last Learning Rate: {:4f}'.format(scheduler.get_last_lr()[0]))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

"""
This methods takes a model and trains and validates on the validation set for each epoch.
Each epoch checks if the current model is the best model in terms of its validation accuracy.
If so, the model is saved and returned. The loss function is the eenrgy-based loss function

model - the architecture (e.g. densenet)
dataloaders - 
criterion - the loss function
optimizer - the alogirthm to optimize (e.g. SGD)
num_epochs - how many epochs to train for
is_inception - indicates this model is not inception
"""
def train_model_energy(model, dataloaders, dataloader_out, optimizer, num_epochs,
                scheduler, device, args):

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Detect if we have a GPU available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        """
        Train
        """
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, inputs_out in zip(dataloaders['train'], dataloader_out):
            
            data = torch.cat((inputs[0], inputs_out[0]), 0)
            labels = inputs[1]
            data, labels = data.cuda(), labels.cuda()
            
            # forward
            torch.set_grad_enabled(True)
            outputs = model(data)
            _, preds = torch.max(outputs[:len(inputs[0])], 1)

            # backward
            
            optimizer.zero_grad()

            loss = F.cross_entropy(outputs[:len(inputs[0])], labels)
            Ec_out = -torch.logsumexp(outputs[len(inputs[0]):], dim=1)
            Ec_in = -torch.logsumexp(outputs[:len(inputs[0])], dim=1)
            loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())
      

            loss.backward()
            optimizer.step()
            scheduler.step()

            # statistics
            running_loss += loss * inputs[0].size(0)
            running_corrects += torch.sum(preds == labels.data)
            
           
            
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
             'train', epoch_loss, epoch_acc))
        print('Last Learning Rate: {:4f}'.format(scheduler.get_last_lr()[0]))
        
        """
        val
        """
        model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over data.
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # Get model outputs
            torch.set_grad_enabled(False)
            outputs = model(inputs)
                
            loss = F.cross_entropy(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)   

        epoch_loss = running_loss / len(dataloaders['val'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_loss, epoch_acc))
        print('Last Learning Rate: {:4f}'.format(scheduler.get_last_lr()[0]))

            # deep copy the model
        if  epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

"""
This helper function sets the .requires_grad attribute of the parameters in the model
to False when we are feature extracting. By default, when we load a pretrained model 
all of the parameters have .requires_grad=True, which is fine if we are training 
from scratch or finetuning. However, if we are feature extracting and only want
to compute gradients for the newly initialized layer then we want all of the other 
parameters to not require gradients. 
"""
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


"""
This method reshapes the last layer to fit the number of classes, the default
model has 1000 classes. 

model_name - name of the model
num_classes - name of the classes
feature_extract - whether we are feature extracting or not
use_pretrained - if using a pre-trained model or random weights
"""
def initialize_model(model_name, num_classes, feature_extract, use_pretrained):

    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    # DenseNet
    if model_name == "densenet":

        model_ft = models.densenet121(pretrained=use_pretrained)

        # Requires to set params as requires_grad = False first, then the linnear
        #   classifier is set up with the default requires_grad = True so only
        #   this last layer is trainable.
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "wide":
        """ WideResNet50-2
        """
        model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("This method only looks for \"densenet\" or \"resnet\", if you are trying another "
              + "model, go on pytorch transfer learning tutorial and add the required"
              + "code for that model")
        exit()

    return model_ft, input_size


"""
This method takes a model and predicts on the test data, the total accuracy
over the predictions is returned

model- the model to do the prediction
dataloaders - contains the test data
"""
def test_model(model, dataloaders):


    since = time.time()

    # Detect if we have a GPU available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()   # Set model to evaluate mode

    running_corrects = 0

    # Iterate over test data.
    for inputs, labels in dataloaders['test']:
   
        #feed Forward the inputs
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    total_acc = running_corrects.double() / len(dataloaders['test'].dataset)

    print('{} Acc: {:.4f}'.format('test', total_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return total_acc

"""
This method takes takes in a model, a tensor of images, a tensor of labels, and
tests the model to determine its accuracy prediction, additionally, this method 
has an option for OOD detection

"""
def test_model_custom(model, images, labels, data_list, T, device, logit_mode, top_n):

    print("Testing the model with OOD detection...")
    since = time.time()

    model = model.to(device)
    labels = labels.to(device)
    model.eval()   # Set model to evaluate mode

    running_corrects = 0
    energy_scores = torch.zeros((labels.shape[0]))
    curr_batch = 0

    # feed forward inputs
    for data in data_list:
        
        print("Testing on " + data +" data")
        for img_batch, _ in images[data]:
            
            batch_size = len(img_batch)
            
            img_batch = img_batch.to(device)

            outputs = model(img_batch)
            _, preds = torch.max(outputs, 1)
            
            if logit_mode == 'std_aug':
               # top, _ = torch.topk(outputs, top_n + 1)
               # top_n_mean = (torch.div(torch.sum(top[:,1:], dim = 1), top_n)).unsqueeze(1)
                #distance = top[:,0].unsqueeze(1) - top_n_mean
              #  outputs = torch.add(outputs, distance)
              std = torch.std(outputs, 1)
              outputs = outputs + std[:, None]
                
            """
            NOTE:
            somehow, this commented-out statement keeps creating tensors and accumulating memory,
            using .data 'unpacks' the data from the tensor preventing memory accumulation
            
            energy_scores[curr_batch: curr_batch + batch_size] = -T*torch.logsumexp(outputs/T, dim=1)
            """
            energy_scores[curr_batch: curr_batch + batch_size] = (-T*torch.logsumexp(outputs/T, dim=1)).data 
         
            running_corrects += torch.sum(preds == labels.data[curr_batch: curr_batch + batch_size])
            curr_batch = curr_batch + batch_size
            
       
    total_acc = running_corrects.double() / len(labels)

    print('{} Acc: {:.4f}'.format('test', total_acc))

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    
    return total_acc, energy_scores


"""
This method computes the mean and std of energy scores for each dataset

energy_scores - the collection of energy scores for all datasets
labels - the labels of each row in energy_scores
num_classes_in - number of in-distribution classes
data_list - list of data sets
in_data_name - the name of the data set that is in distribution
"""
def print_stats(energy_scores, labels, num_class_in, data_list, in_data_name):
    
    mean = []
    std = []
    ood_counter = 0
    for dat, data in enumerate(data_list):
        
        if data ==  in_data_name:
            
            indeces = ((labels < num_class_in).nonzero(as_tuple=False)).squeeze(1)
           # data_list[dat] =  data_list[dat] + ' (ID)'
        else:
            curr_label = num_class_in + ood_counter
            indeces = ((labels == curr_label).nonzero(as_tuple=False)).squeeze(1)
           # data_list[dat] =  data_list[dat] + ' (OOD)'
            ood_counter += 1
            
        scores = energy_scores.gather(0, indeces)
        mean.append(torch.mean(scores).item())
        std.append(torch.std(scores).item())
        
    print('data: ')
    print(' '.join(data_list))
    print('mean: ')
    print([str(m) for m in mean])
    print('std: ')
    print([str(s) for s in std])


"""
This method takes in enerhy scores for a set of predictions, and plots them into
a Histogram, where the x axis is energy bins and then y axis is 

energy_scores - a list of lists, each of these list contains the energy score,
                the true label, and the predicition
num_class_in - the number of in-distribution classes
num_class_tot - the number of total classes, in- and out-of-distribution
num_bins - the number of bins for the histogram                 
"""
def visualize_energy_scores(energy_scores, labels, num_class_in, data_list, in_data_name ,
                            num_bins, ranges, min_data_size):
    
    colors = ['darkgreen', 'lime', 'greenyellow',
             'gold', 'darkorange', 'orangered',
              'red', 'maroon', 'black']
    
#    colors = ['darkgreen', 'gold', 'red']
   # colors = ['mediumspringgreen', 'crimson']
    ood_counter = 0
    for dat, data in enumerate(data_list):
        
       if data ==  in_data_name:
           """
           labels < num_class_in, the current implementation is to check breas or lymph
           """
           #indeces = ((labels == num_class_in-1).nonzero(as_tuple=False)).squeeze(1)
           indeces = ((labels < num_class_in).nonzero(as_tuple=False)).squeeze(1)
           
       else:
           curr_label = num_class_in + ood_counter
           indeces = ((labels == curr_label).nonzero(as_tuple=False)).squeeze(1)
           ood_counter += 1
       
       scores = energy_scores.gather(0, indeces)
       scores = scores[0:min_data_size]
       scores = scores.numpy()
      
       plt.hist(scores, bins = num_bins[dat], range = ranges[dat], facecolor = "None", edgecolor = colors[dat], alpha=0.5, linewidth = 1.5)
     
    plt.xlabel('Energy Scores')
    plt.ylabel('Counts')
    plt.title(r'Histogram for Energy Score Counts')
    plt.legend(data_list)     
    plt.show()
       
      
"""
gets the FPR for an OOD dataset at the TPR of 95%. This method finds a threshold
for which 95% of data is flagged correctly, and finds the FPR for each data in data
set and pits the results

energy_scores - the collection energy scores
labels - the collection of labels
num_class_in - the number of in-dist classes
data_list - list of the datasets
in_data_name - name of the ID dataset
"""       
def get_fpr(energy_scores, labels, num_class_in, data_list, in_data_name):
   
    # Ge the in-distribution energy scores
    indeces = ((labels < num_class_in).nonzero(as_tuple=False)).squeeze(1)
    scores = energy_scores.gather(0, indeces)
    # Find the in-dist threshold for 95% tpr
    tpr = torch.tensor([0.95])
    threshold = torch.quantile(scores, tpr)
    print('The treshold: ' + str(threshold.item()))
    FPRs = []
    print('Printing FPR for: ')
    ood_counter = 0
    for dat, data in enumerate(data_list[1:]):
        
        curr_label = num_class_in + ood_counter
        indeces = ((labels == curr_label).nonzero(as_tuple=False)).squeeze(1)
        scores = energy_scores.gather(0, indeces)
        false_positives = (scores <= threshold).nonzero(as_tuple=False)
        fpr = len(false_positives) / len(scores)
        FPRs.append(fpr)
        print(data + ': ' + str("{:.2f}".format(fpr)))
        ood_counter += 1
        
    return FPRs
        
"""
compute the mahalanobis distances between each x in X and the vector of means
mu
"""        
def get_dist(X, logits, preds):
    
    dist = torch.zeros((X.shape[0]))
    mu = torch.mean(logits, 0)
    
    for i,x in enumerate(X):
        #pred = preds[i].item()
        #delta = x - logits[pred, :]
       # x_mu = torch.mean(x)
      #  delta = mu - x_mu
        delta = x - mu
        dist[i] = torch.dot(delta, delta) #delta
  
    return torch.sqrt(dist)   
    #return dist    
        
def test_model_maha(model, images, labels, data_list, T, device):    

    print("Testing the model with OOD detection using Mahalanobis")
    since = time.time()
   
    model = model.to(device)
    model.eval()   # Set model to evaluate mode
    
    # Loading logits
    print('loading logits_e1t5.47')
    logits = torch.load('./models/logits_e1t5.47.pt')

    energy_scores = torch.zeros((labels.shape[0]))
    curr_batch = 0
   
    # feed forward inputs
    for data in data_list:
        
        print("Testing on " + data +" data")
        for img_batch, _ in images[data]:
            
            batch_size = len(img_batch)
            img_batch = img_batch.to(device)
   
            outputs = model(img_batch)
            _, preds = torch.max(outputs, 1)
            
            #distance = (1/len(img_batch))*get_dist(outputs, logits, preds)
            distance = get_dist(outputs, logits, preds)
            distance = distance.to(device)
                    
            outputs = outputs - 0.01*distance[:, None]
                
            """
            NOTE:
            somehow, this commented-out statement keeps creating tensors and accumulating memory,
            using .data 'unpacks' the data from the tensor preventing memory accumulation
            
            energy_scores[curr_batch: curr_batch + batch_size] = -T*torch.logsumexp(outputs/T, dim=1)
            """
            energy_scores[curr_batch: curr_batch + batch_size] = (-T*torch.logsumexp(outputs/T, dim=1)).data 
         
            curr_batch = curr_batch + batch_size
            
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    
    return energy_scores    
           
    
          
     
        
       
    
    