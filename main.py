"""
Created on Wed Mar 12 16:22:22 2025.

@author: kayol
"""
from os import sep, getcwd
from torch import no_grad, optim, manual_seed, linspace, tensor, mean
from torch.cuda import is_available
from torch.nn import Module, Sequential
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from numpy.random import seed

from torchptrbf.layers import vanilla
from torchptrbf.loss import mse_loss, calc_loss_loader
from torchptrbf.datasets import beamforming_dataset_loader

import matplotlib.pyplot as plt

# PT-RBF parameters
ptrbf_arc = {
    'inputs': 6,
    'outputs': [50, 3],
    'neurons': [50, 50],
    'batch': 1,
    'lr': 3e-3,
    'epochs': 20,
    'len_train': 1e4,
    'len_val': 1e3,
    'len_inf': 1e4,
    'random_seed': 123
    }

def plot_losses(num_epochs, train_losses, val_losses):
    """
    Plot training and validation losses against epochs and data seen.

    This function generates a loss curve with two x-axes:
    - The primary x-axis represents epochs.
    - The secondary x-axis represents the number of tokens seen.
    The plot includes both training and validation losses.

    Returns
    -------
        num_epochs (int): Number of training epochs.
        train_losses (list of float): Training losses.
        val_losses (list of float): Validation losses.

    Parameters
    ----------
        None: The function saves the plot as 'loss-plot.pdf' and displays it.
    """
    folder = getcwd() + sep + "results" + sep

    epochs_seen = linspace(0, num_epochs, len(train_losses))

    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    # only show integer labels on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(folder + "loss-plot.pdf")
    plt.show()


def plot_consts(const, name):
    folder = getcwd() + sep + "results" + sep

    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.scatter(tensor(const).real, tensor(const).imag)

    ax1.set_xlabel("I")
    ax1.set_ylabel("Q")
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(folder + name + ".pdf")
    plt.show()


class ptrbf_nn(Module):

    def __init__(self, inputs, neurons, outputs):
        super().__init__()

        input_vec = [inputs] + outputs[:-1]

        self.layers = Sequential(*[vanilla(i, n, o)
                                   for i, n, o in zip(input_vec,
                                                      neurons, outputs)])

    def forward(self, x):

        y = self.layers(x)

        return y


def train(model, train_loader, val_loader, optimizer, device, num_epochs):

    # Initialize lists to track losses.
    train_losses, val_losses = [], []

    # Initialize lists to track losses.
    train_losses_batch, val_losses_batch = [], []

    print('Training PTRBF...')

    # Main training loop.
    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}:")

        # Set model to training mode.
        model.train()

        for input_batch, target_batch in train_loader:

            # Reset loss gradients from previous batch iteration
            optimizer.zero_grad()

            # Pass data to device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Compute prediction error
            pred = model(input_batch)
            loss = mse_loss(target_batch, pred)

            # Calculate loss gradients
            loss.backward()
            # Update model weights using loss gradients
            optimizer.step()

        train_loss, val_loss = evaluate_model(model, train_loader,
                                              val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss {train_losses[-1]:.3f}, Val loss {val_losses[-1]:.3f}")

    return train_losses, val_losses


def evaluate_model(model, train_loader, val_loader, device):

    model.eval()

    with no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()

    return train_loss, val_loss


def inference(inference_loader, model, device):

    model.eval()

    qam4 = []
    psk8 = []
    qam16 = []

    with no_grad():

        for input_batch, _ in inference_loader:

            # Pass data to device
            input_batch = input_batch.to(device)

            # Compute prediction error
            pred = model(input_batch)

            qam4.append(pred[0, 0, 0])
            psk8.append(pred[0, 0, 1])
            qam16.append(pred[0, 0, 2])

    model.train()

    return qam4, psk8, qam16


# Seed of random numbers numpy
seed(ptrbf_arc['random_seed'])

# Create data loaders
train_loader = beamforming_dataset_loader(len_data=ptrbf_arc['len_train'],
                                          batch_size=ptrbf_arc['batch'],
                                          shuffle=True)

val_loader = beamforming_dataset_loader(len_data=ptrbf_arc['len_val'],
                                        batch_size=1)

inf_loader = beamforming_dataset_loader(len_data=ptrbf_arc['len_inf'],
                                        batch_size=1)

# Get device for training
#device = "cuda" if is_available() else "cpu"
device = 'cpu'
print(f"Using {device} device")

# For reproducibility due to the shuffling in the data loader.
manual_seed(ptrbf_arc['random_seed'])

ptrbf = ptrbf_nn(ptrbf_arc['inputs'], ptrbf_arc['neurons'],
                 ptrbf_arc['outputs']).to(device)
ptrbf.eval()


optimizer = optim.SGD(ptrbf.parameters(), lr=ptrbf_arc['lr'])

# Model training
train_losses, val_losses = train(
    ptrbf, train_loader, val_loader, optimizer, device,
    num_epochs=ptrbf_arc['epochs'])

# Plot the training and validation losses.
plot_losses(ptrbf_arc['epochs'], train_losses, val_losses)

# Model inference
qam4, psk8, qam16 = inference(inf_loader, ptrbf, device)

plot_consts(qam4, "QAM-4")
plot_consts(psk8, "PSK-8")
plot_consts(qam16, "QAM-16")
