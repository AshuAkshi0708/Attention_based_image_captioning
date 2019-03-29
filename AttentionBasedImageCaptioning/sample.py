
import torch
from torch.autograd import Variable

import torch.nn.functional as F

# Other libraries for data manipulation and visualization
import os
import sys
import numpy as np
import pickle
import checkpointing
# import metrics
import utils
import pickle

import argparse
from vocab_builder import Vocabulary
from models import *
from torchvision import transforms
from dataloader import *

import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):

    vocab = pickle.load(open("data/vocab.pkl", "rb"))

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    embed_size = 256
    vocab_size = len(vocab)
    hidden_size = 300
    num_layers = 1

    lstm_model = Caption_Generator_LSTM(vocab_size, embed_size, hidden_size, num_layers)
    cnn_model = Feature_Generator_CNN(embed_size)
    model_name = "IC_model"
    model = ImageCaptioningModelWrapper(lstm_model, cnn_model, model_name)
    
    model.load_checkpoints("best")
    model.put_on_gpu(device)
    model.cnn_model.eval()
    model.lstm_model.eval()

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = model.cnn_model.forward(image_tensor)
    sampled_ids = model.lstm_model.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    image = Image.open(args.image)
    plt.imsave(args.image, np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    
    args = parser.parse_args()
    main(args)
