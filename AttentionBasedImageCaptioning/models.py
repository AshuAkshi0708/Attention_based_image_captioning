# CharLevelLSTM is the base class for the model
# All other models are specific instantiations
# Those subclasses can be conveniently used with the models dictionary

import torch
import torch.nn as nn
import torchvision.models as pretrained
import checkpointing
from torch.nn.utils.rnn import pack_padded_sequence

class ImageCaptioningModelWrapper():
  def __init__(self, lstm_model, cnn_model, model_name):
    self.lstm_model = lstm_model
    self.cnn_model = cnn_model
    self.model_name = model_name

  def forward(self, images, captions, lengths):
    cnn_output = self.cnn_model.forward(images)
    generated_captions = self.lstm_model.forward(cnn_output, captions, lengths)

    return cnn_output, generated_captions

  def put_on_gpu(self, computing_device):
    self.lstm_model.to(computing_device)
    self.cnn_model.to(computing_device)

  def save_checkpoints(self, epoch):
    checkpointing.save_checkpoint(self.model_name+"_lstm", self.lstm_model, epoch)
    checkpointing.save_checkpoint(self.model_name+"_cnn", self.cnn_model, epoch)

  def load_checkpoints(self, epoch):
    checkpointing.load_model(self.model_name+"_lstm", self.lstm_model, epoch)
    checkpointing.load_model(self.model_name+"_cnn", self.cnn_model, epoch)

  def get_params(self):
    return list(self.lstm_model.parameters()) + list(self.cnn_model.parameters())


class Feature_Generator_CNN(nn.Module):
  def __init__(self, embed_size):
      super(Feature_Generator_CNN, self).__init__()
      resnet = pretrained.resnet18(pretrained=True)
      # Different CNN models:
      # Resnet 34 and the original inception net
      # resnet = pretrained.resnet34(pretrained=True)
      # resnet = pretrained.googlenet(pretrained=True)
      modules = list(resnet.children())[:-1]      # delete the last fc layer.
      self.resnet = nn.Sequential(*modules)
      self.linear = nn.Linear(resnet.fc.in_features, embed_size)
      self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

  def forward(self, images):
    with torch.no_grad():
        features = self.resnet(images)
    features = features.reshape(features.size(0), -1)
    features = self.bn(self.linear(features))
    return features

  def parameters(self):
    return list(self.linear.parameters()) + list(self.bn.parameters())

class Caption_Generator_LSTM(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers, max_seg_length=20):
      super(Caption_Generator_LSTM, self).__init__()
      self.embed = nn.Embedding(vocab_size, embed_size)
      self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
      self.linear = nn.Linear(hidden_size, vocab_size)
      self.max_seg_length = max_seg_length

  def forward(self, features, captions, lengths):
    embeddings = self.embed(captions)
    embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
    packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
    hiddens, _ = self.lstm(packed)
    outputs = self.linear(hiddens[0])
    return outputs

  def sample(self, features, states=None):
    """Generate captions for given image features using greedy search."""
    sampled_ids = []
    inputs = features.unsqueeze(1)
    for i in range(self.max_seg_length):
        hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
        outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
        _, predicted = outputs.max(1)                        # predicted: (batch_size)
        sampled_ids.append(predicted)
        inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
        inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
    sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
    return sampled_ids
