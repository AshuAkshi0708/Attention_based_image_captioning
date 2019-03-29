# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SequentialSampler

# Other libraries for data manipulation and visualization
import os
import numpy as np 
import pickle as pkl

import math
# Uncomment for Python2
# from __future__ import print_function


class MusicDataset(Dataset):

  def __init__(self, split="train", chunk_size=100):
    self.chunk_size=chunk_size
    with open("encoding_map.pkl", 'rb') as f:
      self.idxOfChar, self.charAtIdx = pkl.load(f)
    in_fname = f"pa4Data/{split}.txt"
    with open(in_fname, 'r') as infile:
      self.input = list(infile.read())


  def __len__(self):
    return math.ceil(len(self.input)/self.chunk_size)
 

  def __getitem__(self, index):
    '''
      defines iteration over chunks of characters. Can be used as follows:

      for chunk in train_loader:
        inputs = chunk[:-1, :]
        targets = chunk[1:, :]
    '''
    start = index * self.chunk_size
    end = min(start + self.chunk_size + 1, len(self.input))
    chars_from_pos = self.input[start:end]
    #chars_from_pos = chars_from_pos + ['EOF'] * ((self.chunk_size + 1) - len(chars_from_pos))
    hot_indices = torch.tensor([self.idxOfChar[c] for c in chars_from_pos]) 
    
    one_hot = torch.zeros(len(chars_from_pos), len(self.idxOfChar.keys()) )
    one_hot[torch.arange(len(chars_from_pos)), hot_indices ] = 1
    return one_hot

    

def create_split_loaders(batch_size=1, chunk_size=100, seed=17, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - extras: (dict) 
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory 
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """
    
    print("Inside create_split_loaders") 
    train_dataset = MusicDataset("train", chunk_size)
    val_dataset = MusicDataset("val", chunk_size)
    test_dataset = MusicDataset("test", chunk_size)
    print("made train, val, test dataset objects") 

    #no shuffling, SequentialSampler

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        print(f"cuda is available: num workers: {num_workers} pin mem: {pin_memory}") 
        
    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=num_workers, pin_memory=pin_memory)

    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)

    
    print(f"Returning from create_split_loaders") 
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)

 
if __name__ == '__main__':
  tr, v, te = create_split_loaders()
  print(len(tr))
  for idx, batch in enumerate(tr):
    print(idx, batch.shape)
