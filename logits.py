"""
Created on Thu Nov 25 12:14:58 2021

Description: This files looks at the logit disitrubtion differences between
             in-distribution and out-of-distribution datset to see if any
             measurable disfferences in the distribution could be used to redefine
             E(x, y) = -f(x)_i to something a bit more robust
             

@author: Juan Rios
"""



#%% Import statements
import os
import time
import torch

import HelperMethods as hp
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

#%% Hyper parameters

data_list =  ['healthy_tissue','in situ']#,'hymnoptera']#, 'rodentia', 'plantae', 'tissue'] #always put ID in index 0
in_data_name = data_list[0] 
data_dir = './data/medical'
model_name = "wide" 
models_dir = './models'
model_Id = 'e2t7.22'
T = 1 # as done in OOD paper, gives least FPR
batch_sz = 15 # bigger batch size might run outta memory
top_n = 5
cls = 0 # pick a class to visualize (0-53)
dim = 1 # which dimension of the logits array to visualize 0 or 1
use_pretrained = False
feature_extract = False

#%% Helper Methods

"""
Get the number of classes under the data_dir/train subdirectory
"""
def get_num_classes():
    
    try:
        files = os.listdir(os.path.join(data_dir, in_data_name, 'train'))
        return len(files)    

    except:
         print("Error: in get_num_classes()")

"""
loads all the images into a data loader, builds a custom labels tensor
and returns the dataloader, the labels, and a list of the datasets in the
dataloader
"""
def collect_images(input_size, num_classes_in):
    
    # initializing the transforms
    data_transforms =  transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    # initializing the dataset    
    print("Initializing datasets and dataloader...")
    
    image_datasets = {data: datasets.ImageFolder(os.path.join(data_dir, data, 'val'), 
                      data_transforms) for data in data_list}
    
    data_loader_dic = {data: torch.utils.data.DataLoader(image_datasets[data],
                        batch_size= batch_sz, shuffle=False, 
                        num_workers=4) for data in data_list}

    return data_loader_dic


"""
This method takes a list od data sets, and returns, logits, a tensor containing
 the average output for each class prediction. logits is a tensor of size
 [num_classes_in x num_classes_in, num_datasets]. num_classes is the number of 
 classes the model can predict. the first 2 dimensions is a square table
 where the row of each table represents the predicted class (highest logit), 
 and each column represents the logit output for all the classes. The third dimension
 indecex each data set.
 
 example: supposed you have 2 datasets, and the model can predict between 3 classes
 then then are 2 square tables:
     
     logits = |0.8 0.1 0.1|   |0.7 0.1 0.2|
              |0.2 0.5 0.3|   |0.1 0.9 0.0|
              |0.0 0.1 0.9| , |0.3 0.3 0.4|
"""
def get_logits(model, images, data_list, T, device, num_classes_in):
    
    print("Getting the logits ...")
    since = time.time()
    
    # Set model to evaluate mode, transfer to device
    model = model.to(device)
    model.eval()   
    # Initialize data structure to hold logits
    logits = torch.zeros((num_classes_in, num_classes_in, len(data_list))).to(device)
    logit_counts = torch.zeros((num_classes_in, num_classes_in, len(data_list))).to(device)

    # feed forward inputs
    for dat_idx, data in enumerate(data_list):
        
        print("Getting logits from " + data +" data")
        
        for img_batch, _ in images[data]:
            
            img_batch = img_batch.to(device)
   
            outputs = model(img_batch)
            _, preds = torch.max(outputs, 1)
            
            for pred_idx, pred in enumerate(preds):
                logits[pred.item(), :, dat_idx] = (logits[pred.item(), :, dat_idx] 
                + outputs[pred_idx]).data
                logit_counts[pred.item(), :, dat_idx] = logit_counts[pred.item(), :, dat_idx]. data + 1
    
    time_elapsed = time.time() - since
    print('Getting logits complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    logits = torch.div(logits, logit_counts + 0.001)
    return logits

def get_logit_distances(logits):
    
    distances = torch.zeros((logits.shape[0], logits.shape[2]))
    
    for cls_i, cls in enumerate(logits):
        
        for dat_i in range(0, logits.shape[2]):
            
            highest_logit = cls[cls_i, dat_i]
            runner_ups = torch.topk(cls[:, dat_i], top_n + 1)
            top_n_mean = (torch.sum(runner_ups[0]) - highest_logit)/top_n
            
            distances[cls_i, dat_i] = highest_logit - top_n_mean
            
    return distances

def visualize_logits(logits, dim):
    
    for dat_i, data in enumerate(data_list):
        
        mu = (torch.mean(logits[:,:, dat_i], dim))
        mu = mu.cpu().numpy()
        sigma = torch.std(logits[:,:, dat_i], dim)
        sigma = sigma.cpu().numpy()
        print('sigma at ' + str(dat_i) + ':' + str(sigma[cls]))
        
        if dim == 0:
            x = logits[:, cls, dat_i]
        else:
            x = logits[cls, :, dat_i]
            
        
        x = x.cpu().numpy()
        x = np.sort(x)
        plt.plot(x, stats.norm.pdf(x, mu[cls], sigma[cls]))
        
    
    plt.xlabel('logits')
    plt.ylabel('P(logits)')
    plt.title(r'Gaussian for logits')
    plt.legend(data_list)     
    plt.show()
    
    print('mean of class logits:')
    print(logits[cls,cls, :])
#%% Main

def main():
    
    # Get the number of in-distribution classes
    num_classes_in = get_num_classes()
    
    # Load the model
        # Load the trained model
    model_ft, input_size = hp.initialize_model(model_name, num_classes_in, 
                                            feature_extract, use_pretrained)
        # Load the dictionary of paramerters
    model_dict = torch.load(os.path.join(models_dir, model_Id))
        # Set the untrained model params to the loaded
    model_ft.load_state_dict(model_dict) 
        # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    # Load the data
    # Get the val images across all datasets and their labels
    images = collect_images(input_size, num_classes_in)
    
    # Get logits
    logits = get_logits(model_ft, images, data_list, T, device, num_classes_in)
    
    # save logits
    #torch.save(logits[:,:,1], 'logits_' + model_Id + '.pt')
    
   #distances = get_logit_distances(logits)
    
    visualize_logits(logits, dim)
    #print(distances)
    
    
    

#%% call to main

"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
  main()

