import torch

# Other libraries for data manipulation and visualization
import os
import pickle




# Copied from PA3
def write_logs(model_name, loss_values, log_phase="train", log_type="loss", directory="logs_results/"):
  if not os.path.exists(directory):
      os.makedirs(directory)
  path = directory + model_name + "_" + log_phase + "_" + log_type + ".txt"
  with open(path, "a") as f:
    f.write(str(loss_values) + "\n")

def write_metrics(model_name, metric_scores, directory="metrics_results/", batchnum=0):
  if not os.path.exists(directory):
      os.makedirs(directory)
  path = directory + model_name +  "_metrics.txt"
  with open(path, "a") as f:
    f.write("batch num:" + str(batchnum) + "\n") 

    for m in metric_scores.keys():
      f.write(str(m) + ": " + str(metric_scores[m]) + "\n")

    f.write("_________________________________________________________________" + "\n")

def save_metrics(model_name, metric_dict, metric_type, directory="logs_results/"):
  if not os.path.exists(directory):
      os.makedirs(directory)
  filename = directory+model_name +"_"+metric_type+"_metrics"
  outfile = open(filename,'wb')
  pickle.dump(metric_dict,outfile)
  outfile.close()


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


def print_metrics(model_name, metric_type="test", directory="logs_results/"):
  filename = directory+model_name +"_"+metric_type+"_metrics"
  infile = open(filename,'rb')
  metrics = pickle.load(infile)

  for metric, vals in metrics.items():
    if metric == 'confusion':
      val_str=matrix_str(vals)
    
    elif metric == 'aggr':
      val_str = str(vals)
    else:
      val_str = '\n'.join([str(v) for v in vals])

    print(f"\n{metric}:\n{val_str}")

