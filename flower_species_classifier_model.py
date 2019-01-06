##### Imports here #####
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import os
import copy
import PIL
import json
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models

# Load the data #

##### Constants for training #####
'''
You can download the data here (https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip)
'''

# file path
data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# number of subprocesses to use for data loading
num_workers = 4

# how many samples per batch to load
batch_size = 64

# list which has mapping of flower names
flower_label_list = None

# initializing mean and standard deviation
mean = [0.485, 0.456, 0.406]
std_dev = [0.229, 0.224, 0.225]

# defining transforms for the training and validation sets
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,
                                                          std_dev)]),
                  'valid': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,
                                                          std_dev)]) }


# loading the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                 for x in ['train', 'valid']}

# defining the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size,
                                             shuffle = True, num_workers = num_workers)
                for x in ['train', 'valid']}

# defining the dataset_sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}


##### Label mapping #####
'''
You'll need to load in a mapping from category label to category name. You can find this in the file cat_to_name.json.
It's a JSON object which you can read in with the json module.
This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.
'''

with open('cat_to_name.json', 'r') as f:
    flower_label_list = json.load(f)


##### Visualize Data #####
'''
Using the below code you can make sure that you've loaded the data properlyself.
'''

def plot_img(image, plt):
    '''
    This function plots the image
    '''
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = (image)[:3,:,:].unsqueeze(0)

    image = image.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    plt.imshow(image)

def show_images(images, labels):
  '''
  This functions iterates over the images and calls plot_img() function to plot image one by one
  '''
  fig = plt.figure(figsize=(20,5))

  for i in np.arange(20):
      x = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])
      plot_img(images[i], x)
      idx = str(labels[i].item())
      x.set_title(flower_label_list[idx])

# obtain a batch of images from training dataset
data = iter(dataloaders['train'])
images, labels = data.next()
print(labels)

# call show_images() function to visualize a set of data
show_images(images, labels)


# Building the model #

##### Building the classifier #####
'''
building the network using resnet152
'''

# loading a pre-trained network
model = models.resnet152(pretrained=True)

# freezing the parameters so that we train the model which was already trained on imagenet datasets
for param in model.parameters():
    param.requires_grad = False

# defining the model using ReLU activations and Dropout
classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(2048, 1024)),
                                ('relu1', nn.ReLU()),
                                ('dropout', nn.Dropout(0.2)),
                                ('fc2', nn.Linear(1024, 500)),
                                ('relu2', nn.ReLU()),
                                ('dropout', nn.Dropout(0.2)),
                                ('fc3', nn.Linear(500, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

# assigning to fully connected layer
model.fc = classifier

# specify loss function
criterion = nn.NLLLoss()

# specify optimizer
optimizer = optim.Adam(model.fc.parameters(), lr= 0.001)

# decay lr by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


##### Checking whether GPU is available #####
'''
Check if GPU is available and move the model to GPU so that computation is made faster.
'''

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

# use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# move model to 'GPU' if available
model.to(device)


# Training the Network #

##### Training #####

def train_model(model, criterion, optimizer, scheduler, num_of_epochs=25):
  '''
  Function to train the model
  '''

  checkpoint_model_name = 'model_skr_fc_1024.pt'
  checkpoint_model = copy.deepcopy(model.state_dict())

  previous_time = time.time()

  # Tracker for best accuracy
  best_acc = 0.0
  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf

  # for loop for epochs
  for epoch in range(num_of_epochs):

      print('\nIteration: ',epoch+1)


      # for loop for different phases
      for phase in ['train', 'valid']:

          # setting model to train() or eval() mode
          if phase == 'train':
            scheduler.step()
            model.train()
          else:
            model.eval()

          # tracker for loss and accuracy
          phase_loss = 0.0
          phase_acc = 0.0

          # for loop for iterating over data
          for data, labels in dataloaders[phase]:

              # moving data, labels to cuda, if it's available
              data, labels = data.to(device), labels.to(device)

              # apply zero gradients
              optimizer.zero_grad()

              # forward - tracking history only for train phase
              with torch.set_grad_enabled(phase == 'train'):

                output = model(data)
                _, preds = torch.max(output, 1)
                #preds.to(device)

                # calculate the loss
                loss = criterion(output, labels)

                # apply back_prop & optimize in `train phase` only
                if phase == 'train':
                  # backward pass
                  loss.backward()
                  # perform a single optimization step
                  optimizer.step()

              # update phase loss & acc
              phase_loss += loss.item() * data.size(0)
              phase_acc += torch.sum(preds == labels.data)

          # update epoch loss & acc
          epoch_loss = phase_loss / dataset_sizes[phase]
          epoch_acc = phase_acc.double() / dataset_sizes[phase]

          print('Epoch: {} \t {} \n Loss: {:.6f} \t Acc: {:.6f}'.format(
                epoch+1,
                phase,
                epoch_loss,
                epoch_acc))

          # saving model for if 'epoch_acc' is greater than 'best_acc'
          '''
          if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            checkpoint_model = copy.deepcopy(model.state_dict())'''
          if phase == 'valid' and epoch_loss <= valid_loss_min and epoch_acc >= best_acc:
            print('True')
            valid_loss_min = epoch_loss
            best_acc = epoch_acc
            checkpoint_model = copy.deepcopy(model.state_dict())

  # calculate time passed
  present_time = time.time() - previous_time

  print('Training completed: {:.0f}m {:0f}sec'.format(present_time//60, present_time%60))
  print('Best Acc: {:6f}'.format(best_acc))

  # loading the best model
  model.load_state_dict(checkpoint_model)

  return model


##### Call train_model() #####
'''
Training this model takes a huge amount of time.
'''

trained_model = train_model(model, criterion, optimizer, scheduler, num_of_epochs=91)


##### Save the checkpoint #####
'''
Now that your network is trained, save the model so you can load it later for making predictions.
'''

def save_model(model, path, architecture):
  '''
  Function to save model
  '''
  torch.save({ 'architecture': architecture,
               'state_dict': model.state_dict(),
               'class_to_idx': model.class_to_idx },
                path)

# defining the checkpoint filename
 checkpoint_model_name = 'resnet152_trained_model_skr.pt'


##### Saving the trained model locally #####

trained_model.class_to_idx = image_datasets['train'].class_to_idx
save_model(trained_model, checkpoint_model_name, architecture = 'resnet152')

##### Loading the checkpoint #####

def load_model(checkpoint_path):
  '''
  Function that loads a checkpoint and rebuilds the model
  '''

  checkpoint = torch.load(checkpoint_path)

  if checkpoint['architecture'] == 'resnet152':
    model = models.resnet152(pretrained=True)

    # Freezing the parameters
    for param in model.parameters():
        param.requires_grad = False


  else:
    print('Wrong Architecture!')
    return None

  model.class_to_idx = checkpoint['class_to_idx']

  classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(2048, 1024)),
                                ('relu1', nn.ReLU()),
                                ('dropout', nn.Dropout(0.2)),
                                ('fc2', nn.Linear(1024, 500)),
                                ('relu2', nn.ReLU()),
                                ('dropout', nn.Dropout(0.2)),
                                ('fc3', nn.Linear(500, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))


  model.fc = classifier

  model.load_state_dict(checkpoint['state_dict'])

  return model

# call load_model() function
model_loaded = load_model(checkpoint_model_name)


##### Constants for testing #####
'''
Constants means these variables are not changed after it's declared.
Code for testing the trained model goes here
'''

# test dir name
test_dir = 'test'

# tansform for test data
test_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

# test dataset using ImageFolder
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

# load the data from dataset using DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


##### Test with test_loader #####

def load_testing(model_loaded, test_loader):
    '''
    This function uses model_loaded and iterate over test data and prints the test loss and test accuracy
    '''

    test_loss = 0.0
    flower_correct = list(0. for i in range(102))
    flower_total = list(0. for i in range(102))

    # Setting to evaluation mode
    model_loaded.eval()

    model_loaded.to(device)

    criterion = nn.CrossEntropyLoss()

    for data, labels in test_loader:

      model_loaded.eval()
      # moving data, labels to cuda, if it's available
      data, labels = data.to(device), labels.to(device)

      # forward pass - compute predicted outputs by passing inputs to the model
      output = model_loaded.forward(data)

      # calculate the loss
      loss = criterion(output, labels)

      # update test loss
      test_loss += loss.item() * data.size(0)

      # convert output probabilities to predicted class
      _, preds = torch.max(output, 1)
      #preds.to(device)

      # compare predictions to true label
      correct_t = preds.eq(labels.data.view_as(preds))
      correct = np.squeeze(correct_t.numpy()) if not train_on_gpu else np.squeeze(correct_t.cpu().numpy())

      for i in range(len(labels.data)):
          label = labels.data[i]
          flower_correct[label] += correct[i].item()
          flower_total[label] += 1

    # calculate and print avg test loss
    print('Test Loss: {:.6f}\n'.format(test_loss/len(test_loader.dataset)))


    for i in range(102):
      if flower_total[i] > 0:
          print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                  i, 100 * flower_correct[i] / flower_total[i],
                  np.sum(flower_correct[i]), np.sum(flower_total[i])))
      else:
          print('Test Accuracy of %5s: N/A (no training examples)' % (i))


    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(flower_correct) / np.sum(flower_total),
        np.sum(flower_correct), np.sum(flower_total)))

'''
model_loaded = trained_model
Uncomment this if you are not loading the saved model for testing with test data.
'''


##### Call load_testing() #####
load_testing(model_loaded, test_loader)


##### Inference for classification #####

def process_image(image_path):
    '''
    This function processes a PIL image for use in a PyTorch model. Scales, crops,
    and normalizes a PIL image for a PyTorch model, returns an Numpy array.
    '''
    # Load Image an open the image
    image = Image.open(image_path)

    width = image.size[0]
    height = image.size[1]

    # Setting minimum side to 256
    if width > height:
      image = image.resize((width, 256))
    else:
      image = image.resize((256, height))

    left_margin = (image.width - 224) / 2
    lower_margin = (image.height - 224) / 2
    upper_margin = lower_margin + 224
    right_margin = left_margin + 224

    image = image.crop((left_margin, lower_margin, right_margin, upper_margin))

    # normalize
    image_arr = np.array(image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std_dv = np.array( [0.229, 0.224, 0.225])

    image_arr = (image_arr - mean)/std_dv
    image_arr = image_arr.transpose((2, 0, 1))

    return image_arr

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# give path of any flower image
image_path = 'test/102/image_08004.jpg'

# process the image and load it as Numpy array
image = process_image(image_path)

# display the image
imshow(image)


##### Class Prediction #####

def predict(image_path, model, top_k=5):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    Implement the code to predict the class from an image file
    '''
    image = process_image(image_path)

    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)

    image_t = image_tensor.unsqueeze_(0)

    # get the predictions
    preds = model.forward(image_t)

    # calculate its probabilities by applying softmax
    probs = F.softmax(preds, dim=1)

    # get top 5 along the column
    top_probs, top_labels = probs.topk(top_k, dim=1)

    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labels = top_labels.detach().numpy().tolist()[0]

    # convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    # get top 5 labels as a list
    top_labels_list = [idx_to_class[label] for label in top_labels]

    # get top 5 flower names as a list
    top_flowers_list = [flower_label_list[ idx_to_class[label] ] for label in top_labels]

    return top_probs, top_labels, top_flowers_list

# get the probabilities and classes to which those flowers belong
probs, classes, _ = predict(image_path, trained_model.to('cpu'))
print('Probs: ',probs)
print('Classes',classes)


##### Sanity Checking #####
'''
Now that we used a trained model for predictions, let us check to make sure it makes sense.
Even if the validation accuracy is high, it's always good to check that there aren't obvious bugs.
We use matplotlib to plot the probabilities for the top 5 classes as a bar graph, along with the input image.
'''

def sanity_check(image_path, model):
  '''
  Function to display an image along with the top 5 classes
  '''
  #set up plot
  fig = plt.figure(figsize = (6,10))
  ax = plt.subplot(2, 1, 1)

  flower_num = image_path.split('/')[1]
  title_ = flower_label_list[flower_num]

  image = process_image(image_path)
  imshow(image, ax, title = title_)

  probs, labels, flower_names = predict(image_path, model)

  plt.subplot(2, 1, 2)
  sns.barplot(x = probs, y = flower_names, color=sns.color_palette()[0])
  plt.show()

# Comment this if your are using a loaded model "model_loaded"
sanity_check(image_path, trained_model.to('cpu'))

# Comment this if your are using a trained model "trained_model"
sanity_check(image_path, model_loaded.to('cpu'))
