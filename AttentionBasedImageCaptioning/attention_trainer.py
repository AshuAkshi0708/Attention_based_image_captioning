import torch
from torch.autograd import Variable

import torch.nn.functional as F

# Other libraries for data manipulation and visualization
import os
import sys
import numpy as np
import pickle
import checkpointing
import metrics
import utils
import pickle

from vocab_builder import Vocabulary
from models import *
from attention_model import *
from torchvision import transforms
from dataloader import *

def IntegerEncoder(captions):
    with open("word_dict.pkl","rb") as f:
        word_dict = pickle.load(f)

    caption = captions[0]
    #caption = min(captions)
    wordIndex = [word_dict[word] for word in caption]
    wordIndex = [0] + wordIndex + [len(word_dict)+1]
    return np.array(wordIndex)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class Trainer(object):

  def __init__(self, model, train_loader, val_loader, loss=None, alpha=.005, epochs=40, batch_size=16):

      self.setup_gpu()
      self.model = model
      model.put_on_gpu(self.computing_device)

    # initializing the loss
      if loss is not None:
        self.criterion = loss
      else:
        self.criterion = torch.nn.CrossEntropyLoss()

      seed = np.random.seed(1)

    # Creating dataloaders
      self.batch_size = batch_size
      self.train_loader = train_loader
      self.val_loader = val_loader
    # Setting the optimizer
      self.learning_rate = alpha
      self.optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, self.model.get_params()), alpha)

    # Model configuration
      self.num_epochs = epochs

    # validation hyper-params
      self.patience_epochs = 3



  def setup_gpu(self):
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


  # offset of 0 starts from beginning
  # offset of N starts with epoch N+1
  def execute_pipeline(self, offset=0):
    # Val losses saved for early stopping
    print("Beginning Training!")
    val_losses = []
    best_model_epoch = "0"
    best_val_loss = 9999999
    patience_count = 0
    grad_clip = 0.5
    alpha_c = 0.5

    if offset > 0:
      self.model.load_checkpoints(f"{offset-1}")

    # Begin training procedure
    for i in range(self.num_epochs):
      epoch = offset + i
      N = 50
      N_minibatch_loss = 0.0
      epoch_loss = 0.0

      for minibatch_count, (images, captions, lengths) in enumerate(self.train_loader, 0):

        # ignore the batch with less than 32 images; causing problems with BN somehow, SMH
        if images.shape[0] < 4:
          continue

        images = images.to(self.computing_device)
        captions = captions.to(self.computing_device)
        targets = captions[:,1:]
        #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        # source, target = source.type(torch.LongTensor), target.type(torch.LongTensor)

        cnn_output, decoder_output = self.model.forward(images, captions, lengths)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_output

        scores = pack_padded_sequence(scores,decode_lengths, batch_first = True)[0]
        targets = pack_padded_sequence(targets,decode_lengths,batch_first = True)[0]

        self.optimizer.zero_grad()

        loss = self.criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        loss.backward()
        clip_gradient(self.optimizer, grad_clip)
        self.optimizer.step()


        # Add this iteration's loss to the total_loss
        N_minibatch_loss += loss.item()
        epoch_loss += loss.item()

        if minibatch_count % N == 0:

          self.record_milestone(N_minibatch_loss, minibatch_count, N, epoch)
          N_minibatch_loss = 0.0

      print("Finished", epoch + 1, "epochs of training")
      self.model.save_checkpoints(str(epoch))
      val_loss = self.record_epoch_end(epoch_loss, minibatch_count)

      # store the best model for later test evaluations
      if val_loss < best_val_loss or best_val_loss == -1:
        best_val_loss = val_loss
        best_model_epoch = str(epoch)
        self.model.save_checkpoints("best")
      self.model.save_checkpoints("last")

      # early stopping
      if len(val_losses) > 2 and val_losses[-1] > val_losses[-2]:
        patience_count += 1
      else:
        patience_count = 0
      if patience_count == self.patience_epochs:
          break
      # early stopping ends

    print("Training complete after ", epoch+1, " epochs")

    self.model.load_checkpoints("best")
    # self.eval_model(self.test_loader, "test")

  def record_milestone(self, N_minibatch_loss, minibatch_count, N, epoch):
    # Print the loss averaged over the last N mini-batches
    N_minibatch_loss /= N
    print('Epoch %d, average minibatch %d loss: %.3f' %
    (epoch + 1, minibatch_count, N_minibatch_loss))

    # Add the averaged loss over N minibatches and reset the counter
    utils.write_logs(self.model.model_name, N_minibatch_loss, loss_type="train_NMini")


  def record_epoch_end(self, epoch_loss, minibatch_count):
    epoch_loss /= minibatch_count
    utils.write_logs(self.model.model_name, epoch_loss, loss_type="train")

    val_loss = 0.0
    val_loss = self.eval_model(self.val_loader, "val")
    return val_loss

  def decode_sentences(self, captions):
    sentences = []
    for row in captions:
      sent_l = []
      for w in row:
        word = vocab.idx2word[w.item()]
        sent_l.append(word)
        if word == '<end>':
          break
      sentence = ' '.join(sent_l)
      sentences.append(sentence)
    return sentences

  def eval_model(self, data_loader, eval_type="val"):
    with torch.no_grad():
      self.model.cnn_model.eval()
      self.model.lstm_model.eval()
      loss = 0.0
      for minibatch_count, (images, captions, lengths) in enumerate(data_loader, 0):

        images = images.to(self.computing_device)

        #captions are sorted
        captions = captions.to(self.computing_device)
        targets = captions[:,1:]
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        # source, target = source.type(torch.LongTensor), target.type(torch.LongTensor)

        cnn_output, decoder_output = self.model.forward(images, captions, lengths)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_output
        targets_packed = pack_padded_sequence(targets,decode_lengths,batch_first = True)[0]
        scores_packed = pack_padded_sequence(scores,decode_lengths,batch_first = True)[0]
        temp_loss = self.criterion(scores_packed, targets_packed)

        #print("SCORES")
        print(scores)
        print("CAPS SORTED")
        print(caps_sorted)
        targets_for_scores = caps_sorted[:,1:]
        gens_for_scores = torch.argmax(scores,dim = 2)
        target_sentences = self.decode_sentences(targets_for_scores)
        genned_sentences = self.decode_sentences(gens_for_scores)

        newline='\n'
        print(f"CAPTION SENTENCES:\n{newline.join(target_sentences)}")
        print(f"------------------------------------------------------------")
        print(f"GENERATED SENTENCES:\n{newline.join(genned_sentences)}")

        smetric = {idx: [lines.strip()] for (idx, lines) in enumerate(genned_sentences)}
        tmetric = {idx: [lines.strip()] for (idx, lines) in enumerate(target_sentences)}


        #smetric = {idx: rr for idx, rr in enumerate(genned_sentences)}
        #tmetric = {idx: rr for idx, rr in enumerate(target_sentences)}
        print(f"SMETRIC:\n{smetric}\nTMETRIC:\n{tmetric}")
        metrics_num = metrics.get_metrics(smetric, tmetric)
        #metrics is a dictionary
        utils.write_metrics(self.model.model_name, metrics_num, batchnum=minibatch_count)

        print("DONE WITH METRICCCSSSS")

        # Add this iteration's loss to the total_loss
        loss += temp_loss.item()
      loss /= minibatch_count
    utils.write_logs(self.model.model_name, loss, loss_type=eval_type)
    return loss


  def sample_from_model(self, data_loader, eval_type="sampling"):
    ##TODO
    pass



if __name__ == '__main__':
  # if len(sys.argv) != 2:
  #   print (f"USAGE: python trainer.py MODEL_NAME\nmodel names: {models.keys()}")
  #   sys.exit(1)

  # model_name = sys.argv[1]
  # model = models[model_name]()
  vocab = pickle.load(open("data/vocab.pkl", "rb"))

  transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

  embed_size = 512
  attention_dim = 512
  decoder_dim = 512
  vocab_size = int(len(vocab))
  hidden_size = 256
  num_layers = 1
  batch_size = 128

  train_images = "/datasets/COCO-2015/train2014"
  train_captions = "data/annotations/captions_train2014.json"
  val_images = "/datasets/COCO-2015/val2014"
  val_captions = "data/annotations/captions_val2014.json"

  train_loader = get_loader(train_images, train_captions, vocab, transform, batch_size, subset_size=80000, shuffle=False)
  val_loader = get_loader(val_images, val_captions, vocab, transform, batch_size, subset_size=15000, shuffle=False)

  #TODO pass in the right params
  encoder = Attention_Encoder(14)
  decoder = Attention_Decoder(attention_dim, embed_size, decoder_dim, vocab_size, 512,dropout = 0.1 )
  model_name = "IC_Attention_model_take2"
  checkpointing.load_model("IC_Attention_model_take2_cnn", encoder,"best")
  checkpointing.load_model("IC_Attention_model_take2_lstm", decoder,"best")
  model = ImageCaptioningModelWrapper(decoder, encoder, model_name)

  trainer = Trainer(model, train_loader, val_loader, epochs=40,alpha = 1e-3)
  #trainer.execute_pipeline()
  #testing metrics, using val_loader for now TODO will change to test_loader"
  trainer.eval_model(val_loader, eval_type="test")
  print("IT RAN ALL THE WAY THROUGH!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
