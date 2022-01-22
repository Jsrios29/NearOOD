"""
Created on Tue Oct 12 20:11:05 2021

Description: This File is the main file to train and save a DenseNet into 
the current directory. Training is done on the last few layers.

References:
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html    

known issues: This file will call itself multiple times when iterating through 
each input and label in the dataloader passed to the train_model function in the
HelperMethods file. To fix this, I must investigate another way to iterate through 
the training data so this file does not run multiple times. It can be seen that this
file runs multiple times because many models are saved in one run but there is only
one line of code saving the model

@author: Juan Rios
"""

#%% ----- Import Statements ---------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse

import HelperMethods as hp

#%% ----- Arg Parser-----------------------------------------------------------

parser = argparse.ArgumentParser(description='PyTorch Image Classification')

# Directories
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset, for example: ./data/animals/carnivora')
parser.add_argument('models_dir', metavar='DIR',
                    help='path to models, for example: ./models/')
parser.add_argument('--aux_data_dir', metavar='DIR', default = './data/auxiliary',
                    help='path to auxiliary ood data, for example: ./data/auxiliary/')
# Training params
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                     help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 5)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# Architecture
parser.add_argument('--feature_extract', choices=('True','False'), default = 'False', 
                    help='feature extract, if True, train last layer, if False, train all layers')
parser.add_argument('--model_name', choices=('densenet','resnet', 'wide'), default = 'resnet', 
                    help='Choose a model to train')
parser.add_argument('--use_pretrained', choices=('True','False'), default = 'False',
                    help='If True, model starts with pre-trained weights')
# Training regime
parser.add_argument('--train_mode', choices=('normal','energy'), default = 'energy',
                    help='If normal, regular corss entropy. If energy, use energy-loss')
parser.add_argument('--m_in', type=float, default=-27., 
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--log_aug', choices=('True', 'False'), default = 'False',
                     help='if false, logits will not be augmented, if true logits will be augmented')



#%% ----- Methods -------------------------------------------------------------
"""
Get the number of classes under the data_dir/train subdirectory

data_dir - the path to the data folder
"""
def get_num_classes(data_dir):
    
    try:
        files = os.listdir(data_dir + "/train")
        return len(files)    

    except:
         print("Error: Could not detect 'train' subdirectory in data directory")


"""
Initialize the dictionary of data loader such that:
    
    dataloader_dict['train'] = dataloader(train image_dataset)
    dataloader_dict['val'] = dataloader(val image-dataset)
    
args - contains program args
data_transforms - contains a dictionary of transforms

return dataloaders_dict - the dictionary of dataloaders
"""
def load_data_loaders(args, data_transforms):
    
    print("Initializing Datasets and Dataloaders \n")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), 
                      data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=args.batch_size, shuffle=True, 
                        num_workers=4) for x in ['train', 'val']}
    return dataloaders_dict
    
"""
Load the data transforms into a dictionary

input_size - the expected dimensions of the input images

return data_transforms - the dictionary containing the transforms
"""        
def load_data_transforms(input_size):
    
    # Data augmentation and normalization for training
    # Data normalization for validation only
    data_transforms = {
        'train': transforms.Compose([
           # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms
 

"""
Print the parameters into the console for verification

args - contains the input arguements
"""   

def print_input(args):
    
    print('Initiating ' + args.train_mode + ' training: \n'
          + str(args.epochs) + ' epochs, ' + str(args.batch_size) 
          + ' batch size ' + 'and a learning rate of ' + str(args.lr) 
          + '.\nUsing a ' + args.model_name + '. Feature extraction is '
          + args.feature_extract + ' and pre-training is ' + args.use_pretrained 
          + '\nMin and Mout are ' + str(args.m_in) + ' and ' + str(args.m_out) 
          +'.\nAnd log_aug is ' + args.log_aug)
    
    
#%% ----- Main ----------------------------------------------------------------    
"""
The maind Method

"""
def main(args):
    
    # Determine number of class
    num_classes = get_num_classes(args.data_dir);
    
    # Initialize the model
    model_ft, input_size = hp.initialize_model(args.model_name, num_classes, 
                                               use_feature_extract, use_pretrained)
    # Loading the transforms
    data_transforms = load_data_transforms(input_size)
    # loading the data into a dictionary
    dataloaders_dict = load_data_loaders(args, data_transforms)
   
    # Send the model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using ' + str(device) + '\n')

    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    """
    NOTE: to print the updatable layers use the snippet of code found in the "Create
          an Optimizer section" of the pyTorch transfer learning tutorial in this
          file's header reference section'
    """
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, args.lr, args.momentum)
    
    # setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer_ft, args.epochs*len(dataloaders_dict['train'])
    )

    # Train and evaluate
    if args.train_mode == 'normal':
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        model_ft, hist = hp.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                 args.epochs, scheduler, device)
    elif args.train_mode == 'energy':
        
        dataloader_out = torch.utils.data.DataLoader(datasets.ImageFolder(args.aux_data_dir, data_transforms['val']),
                            batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        model_ft, hist = hp.train_model_energy(model_ft, dataloaders_dict, dataloader_out,  optimizer_ft,
                                 args.epochs, scheduler, device, args)
    
    # Copy the model back to cpu and save the params to folder
    model_Id = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    model_ft = model_ft.to("cpu")
    torch.save(model_ft.state_dict(), os.path.join(args.models_dir, model_Id))
    
    
#%% ----- Call Main -----------------------------------------------------------
"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
  args = parser.parse_args()
  use_feature_extract  = (args.feature_extract == 'True')
  use_pretrained = (args.use_pretrained == 'True')
  log_aug = (args.log_aug == 'True')
  print_input(args)
  main(args)

