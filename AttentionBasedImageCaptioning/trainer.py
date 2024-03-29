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


class Trainer(object):

  def __init__(self, model, train_loader, val_loader, loss=None, alpha=.005, epochs=15, batch_size=32):

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
      self.optimizer = torch.optim.Adam(self.model.get_params(), alpha)

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

    if offset > 0:
      self.model.load_checkpoints(f"{offset-1}")

    # Begin training procedure
    for i in range(self.num_epochs):
      epoch = offset + i
      self.current_epoch = epoch
      N = 50
      N_minibatch_loss = 0.0
      epoch_loss = 0.0

      for minibatch_count, (images, captions, lengths) in enumerate(self.train_loader, 0):

        # ignore the batch with less than 32 images; causing problems with BN somehow, SMH
        if images.shape[0] < 32:
          continue

        images = images.to(self.computing_device)
        captions = captions.to(self.computing_device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        # source, target = source.type(torch.LongTensor), target.type(torch.LongTensor)

        cnn_output, generated_captions = self.model.forward(images, captions, lengths)

        self.optimizer.zero_grad()
        loss = self.criterion(generated_captions, targets)
        loss.backward()
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
    
    self.current_epoch = "best"
    self.model.load_checkpoints("best")
    self.eval_model(self.val_loader, "test")

  def record_milestone(self, N_minibatch_loss, minibatch_count, N, epoch):
    # Print the loss averaged over the last N mini-batches
    N_minibatch_loss /= N
    print('Epoch %d, average minibatch %d loss: %.3f' %
    (epoch + 1, minibatch_count, N_minibatch_loss))

    # Add the averaged loss over N minibatches and reset the counter
    utils.write_logs(self.model.model_name, N_minibatch_loss, log_phase="train_NMini")


  def record_epoch_end(self, epoch_loss, minibatch_count):
    epoch_loss /= minibatch_count
    utils.write_logs(self.model.model_name, epoch_loss, log_phase="train")

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


  def eval_model(self, data_loader, eval_type="val",eval_criterion = "bleu4"):
    with torch.no_grad():
      loss = 0.0
      all_predictions = []
      all_targets = []
      for minibatch_count, (images, captions, lengths) in enumerate(data_loader, 0):

        images = images.to(self.computing_device)
        captions = captions.to(self.computing_device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        # source, target = source.type(torch.LongTensor), target.type(torch.LongTensor)

        cnn_output, generated_captions = self.model.forward(images, captions, lengths)

        temp_loss = self.criterion(generated_captions, targets)

        #print(f"CAPTIONS: {captions}")
        # Convert to text sentences for metrics

        sampled_words = self.model.lstm_model.sample(cnn_output)
        #print(f"SAMPLED WORDS: {type(sampled_words)} {sampled_words.size()}")

        target_sentences = self.decode_sentences(captions)
        genned_sentences = self.decode_sentences(sampled_words)

        all_predictions.extend(target_sentences)
        all_targets.extend(genned_sentences)

        newline='\n'
        #print(f"CAPTION SENTENCES:\n{newline.join(target_sentences)}")
        #print(f"------------------------------------------------------------")
        #print(f"GENERATED SENTENCES:\n{newline.join(genned_sentences)}")

        #print("DONE WITH METRICCCSSSS")
        # Add this iteration's loss to the total_loss
        loss += temp_loss.item()
      loss /= minibatch_count

    smetric = {idx: [lines.strip()] for (idx, lines) in enumerate(all_predictions)}
    tmetric = {idx: [lines.strip()] for (idx, lines) in enumerate(all_targets)}

    metrics_num = metrics.get_metrics(smetric, tmetric)
    #metrics is a dictionary
    
    utils.write_metrics(self.model.model_name, metrics_num, batchnum=self.current_epoch)
    utils.write_logs(self.model.model_name, loss, log_phase=eval_type)
    utils.write_logs(self.model.model_name, metrics_num["Bleu_4"], log_phase=eval_type, log_type="bleu_metr")

    if eval_criterion == "bleu4":
        return 1-metrics_num["Bleu_4"]
    else:
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

        # If we want to try grayscale, then enable the following line
        # 3 channel grayscale allows grayscaling without messing with required
        # number of input channels for the pretrained CNN

        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

  embed_size = 256
  vocab_size = len(vocab)
  hidden_size = 300
  num_layers = 2
  batch_size = 32

  train_images = "/datasets/COCO-2015/train2014"
  train_captions = "data/annotations/captions_train2014.json"
  val_images = "/datasets/COCO-2015/val2014"
  val_captions = "data/annotations/captions_val2014.json"

  train_loader = get_loader(train_images, train_captions, vocab, transform, batch_size, subset_size=80000, shuffle=False)
  val_loader = get_loader(val_images, val_captions, vocab, transform, batch_size, subset_size=40000, shuffle=False)

  lstm_model = Caption_Generator_LSTM(vocab_size, embed_size, hidden_size, num_layers)
  cnn_model = Feature_Generator_CNN(embed_size)
  #checkpointing.load_model("IC_model_cnn", cnn_model,"best")
  #checkpointing.load_model("IC_model_lstm", lstm_model,"best")
  model_name = "IC_model_lay2_val_metr"
  model = ImageCaptioningModelWrapper(lstm_model, cnn_model, model_name)



  trainer = Trainer(model, train_loader, val_loader, epochs=40)
  trainer.execute_pipeline()
  
  #checkpointing.load_model("IC_model_lay2_cnn", cnn_model,"best")
  #checkpointing.load_model("IC_model_lay2_lstm", lstm_model,"best")
  #trainer.current_epoch = "best"
  #trainer.eval_model(val_loader, eval_type="test")
