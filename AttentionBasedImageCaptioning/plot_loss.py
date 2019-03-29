import numpy as np
import matplotlib.pyplot as plt


def plot_losses(directory, model_name):

    plt.rcParams["lines.linewidth"] = 1.25
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["errorbar.capsize"] = 1.0

    train_losses= np.loadtxt(directory+model_name+"train_loss.txt")
    val_losses = np.loadtxt(directory+model_name+"val_loss.txt")
    epochs = np.arange(1,len(train_losses)+1,1)

    plt.plot(epochs,train_losses,"b",label = "Training Loss")
    plt.plot(epochs,val_losses,"r",label = "Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.title("Training and Validation Losses for the Fiducial Model")

    plt.show()
    plt.savefig(directory+model_name+"losses.pdf")

def compute_metrics(directory, model_name):
    pass


if __name__ == "__main__":
    plot_losses("logs_results/","IC_model_")




