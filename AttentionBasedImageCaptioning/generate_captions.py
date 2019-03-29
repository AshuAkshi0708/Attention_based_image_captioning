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

from vocab_builder import Vocabulary
from models import *
from attention_model import *
from torchvision import transforms
from dataloader import *


class Generator(object):
    def __init__(self,model,test_loader,vocab):
        self.setup_gpu(model)
        self.model = model
        self.model.put_on_gpu(self.computing_device)
        self.vocab = vocab
        self.loader = test_loader
    def setup_gpu(self,model):
    # Check if your system supports CUDA
        use_cuda = torch.cuda.is_available()

        # Setup GPU optimization if CUDA is supported
        if use_cuda:
          self.computing_device = torch.device("cuda")
          self.extras = {"num_workers": 1, "pin_memory": True}
          print("CUDA is supported")
        else: # Otherwise, train on the CPU
          self.computing_device = torch.device("cpu")
          self.extras = False
          print("CUDA NOT supported")
        model.cnn_model.eval()
        model.lstm_model.eval()
    def generate_captions(self):


        for minibatch_count,(images,captions,lengths) in enumerate(self.loader,0):
            if minibatch_count < 1:
                print('I am getting an Image!!')
                images = images.to(self.computing_device)
                captions = captions.to(self.computing_device)
                cnn_output = self.model.cnn_model.forward(images) #Returns feature maps
                batch_size = cnn_output.size(0) #This better be 1
                encoder_dim = cnn_output.size(-1)
                cnn_output = cnn_output.view(batch_size,-1,encoder_dim)
                num_pixels = cnn_output.size(1)
                h,c = self.model.lstm_model.init_hidden_state(cnn_output)


                current_word = "<start>"
                caption = "<start>"
                while current_word != "<end>":
                    #print('I am in!!')
                    current_word_encoded = torch.tensor([self.vocab.word2idx[current_word]],dtype = torch.int64).to(self.computing_device)
                    embeddings = self.model.lstm_model.embedding(current_word_encoded)
                    attention_weighted_encoding, alpha = self.model.lstm_model.attention(cnn_output,h)
                    gate = self.model.lstm_model.sigmoid(self.model.lstm_model.f_beta(h))
                    attention_weighted_encoding = gate * attention_weighted_encoding
                    h,c = self.model.lstm_model.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1),(h,c))
                    preds = self.model.lstm_model.fc(self.model.lstm_model.dropout(h))
                    #Sample from the preds softmax
                    current_word_index = torch.argmax(preds, dim = 1)
                    current_word = self.vocab.idx2word[current_word_index.cpu().item()]
                    caption += current_word
                    print(current_word)
                print(caption)
                print(captions.cpu().numpy().shape)
                print("Target",[self.vocab.idx2word[word] for word in list(captions.cpu().numpy()[0])])
            return (images,caption)



