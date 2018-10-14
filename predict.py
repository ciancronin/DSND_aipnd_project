import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable

import json
import argparse

from PIL import Image


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type = str, default = 'flowers/test/2/image_05100.jpg', help = 'test image')
  parser.add_argument('--topk', type = int, default = '5', help = 'select top k classes')
  parser.add_argument('--gpu_mode', type = str, default = True, help = 'train with gpu')
  parser.add_argument('--checkpoint_path', default = 'checkpoint.pth', type = str, help = 'set the checkpoint path')
  parser.add_argument('--category_names', default = 'cat_to_name.json', type = str, dest = 'category_names')
  return parser.parse_args()


def load_checkpoint(filepath):
  # can't get checkpoint to save more than model due to size issues
  checkpoint = torch.load(filepath)
  saved_model = checkpoint['model']
  #saved_optimizer = checkpoint['optimizer']
  #saved_epoch = checkpoint['epoch']

  return saved_model # , saved_optimizer, saved_epoch


def process_image(image):

  # Process a PIL image for use in a PyTorch model
  # Resize to 256
  image = image.resize((256, 256))
  # Center crop to 224
  image = image.crop((16, 16, 240, 240))

  # 0-255 to 0-1
  image = np.array(image)
  image = image/255.0

  # Nomalization
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  image = (image - mean) / std

  # Transpose
  image = np.transpose(image, (2, 0, 1))

  return image.astype(np.float32)


def predict(image_path, model, topk = 5, device = 'cpu'):
  image = Image.open(image_path)
  image = process_image(image)
  image = np.expand_dims(image, 0)
  image = torch.from_numpy(image)

  model.eval()
  input_image = Variable(image).to(device)
  logits = model.forward(input_image)

  probs = F.softmax(logits,dim=1)
  topk = probs.cpu().topk(topk)

  return (e.data.numpy().squeeze().tolist() for e in topk)


def load_json(json_file):
  with open(json_file, 'r') as f:
    loaded_json = json.load(f)

  return loaded_json


def main():
  # pull in arguments and print
  args = parse_args()
  img_path = args.dir
  gpu_mode = args.gpu_mode
  topk = args.topk
  checkpoint_path = args.checkpoint_path
  category_names = args.category_names

  print('Image path:    {}'.format(img_path))
  print('Load model from: {}'.format(checkpoint_path))
  print('GPU mode:    {}'.format(gpu_mode))
  print('TopK:      {}'.format(topk))

  # load model
  # model, optimizer, epochs = load_checkpoint(checkpoint_path)
  model = load_checkpoint(checkpoint_path)
  class_names = model.class_names

  if gpu_mode and torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  print('Current device:  {}'.format(device))
  model.to(device)

  # adding the cat_to_name label mapping via the json load
  cat_to_name = load_json(category_names)

  probs, classes = predict(img_path, model, topk, device)

  flower_names_map = [cat_to_name[class_names[e]] for e in classes]
  for prob, flower_name in zip(probs, flower_names_map):
    print('{:20}: {:.4f}'.format(flower_name, prob))


if __name__ == '__main__':
  main()