import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

import argparse

import os
import copy

def parse_args():
  parser = argparse.ArgumentParser(description = "Trains a network on a dataset of images and saves the model to a checkpoint")
  parser.add_argument('--data_directory', default = 'flowers', type = str, help = 'set the data dir')
  parser.add_argument('--model_name', default = 'vgg16', type = str, help = 'choose the model architecture (vgg16 or resnet18')
  parser.add_argument('--learning_rate', default = 0.001, type = float, help = 'learning rate')
  parser.add_argument('--hidden_sizes', default = [10024, 1024], nargs = '+', type = int, help='list of integers, the sizes of the hidden layers')
  parser.add_argument('--epochs', default = 3, type = int, help = 'number of epochs to train the model')
  parser.add_argument('--gpu_mode', default = True, type = bool, help = 'set the gpu mode')
  parser.add_argument('--checkpoint_path', default = 'checkpoint.pth', type = str, help='name of checkpoint file to save the model in')
  args = parser.parse_args()
  return args


def train_model(dataloaders, image_datasets, model, criterion, optimizer, epochs = 5, device = 'cpu'):
  print_every = 50
  steps = 0

  for e in range(epochs):
      model.train()
      running_loss = 0
      for ii, (images, labels) in enumerate(dataloaders[0]):
          steps += 1

          images = images.to(device)

          labels = labels.to(device)

          optimizer.zero_grad()

          outputs = model.forward(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          if steps % print_every == 0:
              model.eval()
              valid_loss = 0
              accuracy = 0
              with torch.no_grad():
                  for images, labels in dataloaders[2]:
                      images = images.to(device)
                      labels = labels.to(device)
                      output = model.forward(images)
                      valid_loss += criterion(output, labels).item()

                      probs = torch.exp(output)
                      equality = (labels.data == probs.max(dim=1)[1])
                      accuracy += equality.type(torch.FloatTensor).mean()
              print("Epoch: {}/{}".format(e+1, epochs),
                    "Training Loss: {:.4f}".format(running_loss/len(image_datasets[0])), 
                    "Validation Loss: {:.3f}".format(valid_loss/len(dataloaders[2])),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders[2])))
  return model


def return_classifier(input_size, hidden_sizes, output_size):
  classifier = nn.Sequential()
  if hidden_sizes == None:
      classifier.add_module('fc1', nn.Linear(input_size, output_size))
  else:
      hidden_layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
      classifier.add_module('fc1', nn.Linear(input_size, hidden_sizes[0]))
      classifier.add_module('relu1', nn.ReLU())
      classifier.add_module('drop1', nn.Dropout(.5))
      for i, (layer_one, layer_two) in enumerate(hidden_layer_sizes):
          classifier.add_module('fc' + str(i + 2), nn.Linear(layer_one, layer_two))
          classifier.add_module('relu' + str(i + 2), nn.ReLU())
          classifier.add_module('drop' + str(i + 2), nn.Dropout(.5))
      classifier.add_module('fc_last', nn.Linear(hidden_sizes[-1], output_size))
      classifier.add_module('output', nn.LogSoftmax(dim = 1))
      
  return classifier


def validation(model, dataloader, criterion, device = 'cpu'):
  test_loss = 0
  accuracy = 0
  model.eval()
  for images, labels in dataloader:

      images, labels = images.to(device), labels.to(device)

      output = model.forward(images)
      test_loss += criterion(output, labels).item()

      probs = torch.exp(output)
      equality = (labels.data == probs.max(dim=1)[1])
      accuracy += equality.type(torch.FloatTensor).mean()
  
  model.train()
  
  return test_loss, accuracy


def main():
  args = parse_args()
  data_directory = args.data_directory
  model_name = args.model_name
  learning_rate = args.learning_rate
  hidden_sizes = args.hidden_sizes
  gpu_mode = args.gpu_mode
  epochs = args.epochs
  checkpoint_path = args.checkpoint_path

  print('Data Directory:      {}'.format(data_directory))
  print('Model Architecture:  {}'.format(model_name))
  print('GPU Mode:            {}'.format(gpu_mode))
  print('Epochs:              {}'.format(epochs))
  print('Checkpoint Path:     {}'.format(checkpoint_path))

  train_dir = data_directory + '/train'
  valid_dir = data_directory + '/valid'
  test_dir = data_directory + '/test'

  # TODO: Define your transforms for the training, validation, and testing sets
  train_transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])

  transform_test_val = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

  image_datasets = [datasets.ImageFolder(train_dir, transform=train_transform),
                    datasets.ImageFolder(test_dir, transform=transform_test_val),
                    datasets.ImageFolder(valid_dir, transform=transform_test_val)]

  batch_size = 16
  dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size = batch_size, shuffle=True),
                 torch.utils.data.DataLoader(image_datasets[1], batch_size = batch_size),
                 torch.utils.data.DataLoader(image_datasets[2], batch_size = batch_size)]

  # get the class_names for the training images dataset
  class_names = image_datasets[0].classes

  if gpu_mode and torch.cuda.is_available():
      device = torch.device('cuda')
  else:
      device = torch.device('cpu')

  if model_name == 'vgg16':
      model = models.vgg16(pretrained = True)
      num_in_features = model.classifier[0].in_features
      print(model.classifier)
  elif model_name == 'resnet18':
      model = models.resnet18(pretrained = True)
      num_in_features = model.fc.in_features
      print(model.fc)
  else:
      print("Unknown model, options are 1) 'vgg16' or 2) 'resnet18'")

  # Freeze parameters so we don't backprop through them
  for param in model.parameters():
      param.requires_grad = False

  classifier = return_classifier(num_in_features, hidden_sizes, 102)

  if model_name == 'vgg16':
      model.classifier = classifier
      optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
      criterion = nn.NLLLoss()
  elif model_name == 'resnet18':
      model.classifier = classifier
      optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
      criterion = nn.NLLLoss()
  else:
      print("Unknown model, options are 1) 'vgg16' or 2) 'resnet18'")

  model.to(device)

  # training the model
  model = train_model(dataloaders, image_datasets, model, criterion, optimizer, epochs, device)

  # testing the model
  model.eval()
  with torch.no_grad():
      test_loss, accuracy = validation(model, dataloaders[1], criterion, device = 'cuda')
  print("Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders[1])))
  print("Test Accuracy: {:.3f}".format(accuracy/len(dataloaders[1])))

  # save the model
  model.class_to_idx = image_datasets[0].class_to_idx
  model.class_names = class_names

  # due to size constraints, just going to save the model as saving the optimizer makes the project fail
  checkpoint = {'model': model}
  #               ,
  #               'optimizer': optimizer.state_dict(),
  #               'epoch': epochs}

  torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
  main()