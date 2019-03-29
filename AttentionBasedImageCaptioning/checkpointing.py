import torch

# Other libraries for data manipulation and visualization
import os

def save_general_checkpoint(name, item, epoch, directory="checkpoints/"):
  if not os.path.exists(directory):
      os.makedirs(directory)
  save_path = directory + name + "_" + epoch
  torch.save(item, save_path)

def load_general(name, epoch, directory="checkpoints/"):
  if not os.path.exists(directory):
      return
  load_path = directory + name + "_" + epoch
  return (torch.load(load_path))

def save_checkpoint(model_name, model, epoch, directory="checkpoints/"):
  if not os.path.exists(directory):
      os.makedirs(directory)
  save_path = directory + model_name + "_" + epoch
  torch.save(model.state_dict(), save_path)

def load_model(model_name, model, epoch, directory="checkpoints/"):
  if not os.path.exists(directory):
      return
  load_path = directory + model_name + "_" + epoch
  model.load_state_dict(torch.load(load_path))
