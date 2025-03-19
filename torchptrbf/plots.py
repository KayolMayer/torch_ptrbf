"""
Created on Wed Mar 19 16:03:37 2025.

@author: kayol
"""

from torch import linspace, tensor
from os import sep, getcwd
from matplotlib.pyplot import subplots, savefig, show


def plot_losses(num_epochs, train_losses, val_losses, fig_name):
    """
    Plot training and validation losses against epochs and data seen.

    This function generates a loss curve with two x-axes:
    - The primary x-axis represents epochs.
    - The secondary x-axis represents the number of tokens seen.
    The plot includes both training and validation losses.

    Parameters
    ----------
        num_epochs (int): Number of training epochs.
        train_losses (list of float): Training losses.
        val_losses (list of float): Validation losses.
        fig_name (str): The filename (without extension) to save the figure.

    Returns
    -------
        None: The function saves the plot and displays it.
    """
    folder = getcwd() + sep + "results" + sep

    epochs_seen = linspace(0, num_epochs, len(train_losses))

    fig, ax1 = subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    fig.tight_layout()  # Adjust layout to make room
    savefig(folder + fig_name + ".pdf")
    show()


def plot_consts(const, fig_name):
    """
    Plot a scatter plot of complex-valued data points in the I-Q plane.

    This function takes a list or tensor of complex numbers and plots their
    real (I) and imaginary (Q) components on a 2D scatter plot. The figure
    is saved as a PDF file in the "results" directory.

    Parameters
    ----------
        const (list, torch.Tensor, or numpy.ndarray): A collection of complex
                                                      numbers to be plotted.
        fig_name (str): The filename (without extension) to save the figure.

    Returns
    -------
        None: The function saves the plot and displays it.
    """
    folder = getcwd() + sep + "results" + sep

    fig, ax1 = subplots(figsize=(5, 3))

    ax1.scatter(tensor(const).real, tensor(const).imag)

    ax1.set_xlabel("I")
    ax1.set_ylabel("Q")
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    fig.tight_layout()  # Adjust layout to make room
    savefig(folder + fig_name + ".pdf")
    show()
