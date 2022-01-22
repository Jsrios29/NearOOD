"""
Created on Tue Oct 19 20:46:25 2021

Description: this file loads in a pre-trained model, along with test data for 
             testing and OOD detection.
             The test data must be contained in one folder, where the labels are
             derived from the image file name. For example, the file '0ants.jpg'
             is the first image of the 'ants' class
             

@author: Juan Rios
"""


#%% ----- Import Statements ---------------------------------------------------
import torch
import os
from HelperMethods import initialize_model
from HelperMethods import test_model
from HelperMethods import test_model_custom
from HelperMethods import visualize_energy_scores
from HelperMethods import print_stats
from HelperMethods import get_fpr
from HelperMethods import test_model_maha

from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt


#$%% ----- Hyper Parameters ---------------------------------------------------
data_list = ['healthy_tissue', 'in situ', 'benign', 'invasive']

"""
data_list = ['carnivora', 'artiodactyla', 'accipitriformes', 'hymnoptera'
             , 'tissue'] #always put ID in index 0
num_bins = [25, 25, 25, 25, 25]
min_data_size = 540 # the smallest dataset being visualized, 
                    # careful if visualzing one class from in-dist data    
ranges = [[-20, 0],[-20, 0],[-20, 0], [-20, 0], [-20, 0]]
# ranges = [[-200, -20],[-100, -20],[-200, -20]] # good for medical
"""
#data_list =  ['carnivora', 'artiodactyla','primapes','rodentia','accipitriformes',
              #'hymnoptera','plantae','tissue','pixels'] #always put ID in index 0
num_bins = [30, 30, 75, 75, 50, 50, 50, 50, 50]
min_data_size = 504 # the smallest dataset being visualized, 
                    # careful if visualzing one class from in-dist data    
ranges = [[-60, 0],[-60, 0],[-20, 0],
          [-20, 0],[-80, 0],[-80, 0],
          [-80, 0],[-80, 0],[-80, 0]]

in_data_name = data_list[0] 
data_dir = './data/medical'
model_name = "wide" 
models_dir = './models'
model_Id = 'e2t7.22'
T = 1 # as done in OOD paper, gives least FPR
batch_sz = 15 # bigger batch size might run outta memory

# logit modes: std_aug: use for normally train to add std to log, warning: intra-batch dependance
#              dist: compare the distances to the disitrubtion of logit means derived from val
#              none: when no other mode is wanted
logit_mode = 'none'
top_n = 5


use_pretrained = False
feature_extract = False


#%% ----- Method Definitions --------------------------------------------------

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
    
    labels = torch.zeros(0)
    ood_count = 0
  
    for data_idx, data in enumerate(image_datasets):
        
        # store label
        if data == in_data_name:
            lbls = torch.tensor(image_datasets[data].targets)
        else:
            lbls = (torch.full((len(image_datasets[data]),1), num_classes_in + ood_count)).squeeze(1)
            ood_count += 1
            
        labels = torch.cat((labels, lbls), 0)
        
        
    
    return data_loader_dic, labels.int()


def visualize_fpr(fprs):
   # colors = ['darkgreen', 'lime', 'greenyellow',
        #     'gold', 'darkorange', 'orangered',
             # 'red', 'maroon', 'black']
             
    colors = ['mediumspringgreen', 'palegreen', 'gold', 'darkorange',
              'orangered', 'crimson', 'mediumvioletred' , 'midnightblue', 'black']         
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.5,1.5])
    ax.bar(data_list[1:], fprs, color = colors, width = 0.5)

    plt.xlabel('Datasets')
    plt.ylabel('FPR95')
    plt.title(r'FPR95 for each dataset of increasing distance')  
    plt.show()

  

#%% ----- Call to Main -----
"""
Run the main methodology
"""
def main():
    
    # Get the number of in-distribution classes
    num_classes_in = get_num_classes()
    
    # Load the trained model
    model_ft, input_size = initialize_model(model_name, num_classes_in, 
                                            feature_extract, use_pretrained)
                                                           
    # Load the dictionary of paramerters
    model_dict = torch.load(os.path.join(models_dir, model_Id))
    # Set the untrained model params to the loaded
    model_ft.load_state_dict(model_dict)
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    
    # Get the val images across all datasets and their labels
    images, labels = collect_images (input_size, num_classes_in)
    
    # get the energy scores and results
    if logit_mode == 'dist':
        energy_scores = test_model_maha(model_ft, images, labels, data_list, T, device)
    else:       
        test_results, energy_scores = test_model_custom(model_ft, images, labels, 
                                                    data_list, T, device, logit_mode,
                                                    top_n)
 
    # Visualize as Histogram
    visualize_energy_scores(energy_scores, labels ,num_classes_in, data_list,
                            in_data_name, num_bins, ranges, min_data_size)
    
    print_stats(energy_scores, labels, num_classes_in, data_list, in_data_name)
    
    fprs = get_fpr(energy_scores, labels, num_classes_in, data_list, in_data_name)
               
    visualize_fpr(fprs)
        
#%% ----- Call Main -----------------------------------------------------------
"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
  main()
  # test()
