"""
Created on Wed Mar 19 16:11:59 2025.

@author: kayol
"""

from torch import no_grad
from torch.nn import Module, Sequential
from torchptrbf.layers import vanilla
from torchptrbf.loss import calc_loss_loader


class ptrbf_vanilla(Module):
    """
    Implements a Parametric PTRBF neural network using vanilla layers.

    This model constructs a feedforward network where each layer is initialized
    using a 'vanilla' module. The architecture is defined by the number of
    input neurons, hidden neurons, and output neurons.

    Attributes
    ----------
        layers (torch.nn.Sequential): A sequential container of fully connected
                                      'vanilla' layers.

    Parameters
    ----------
        inputs (int): The number of input features.
        neurons (list of int): A list specifying the number of neurons in each
                               hidden layer.
        outputs (list of int): A list specifying the number of neurons in each
                               output layer.

    Methods
    -------
        forward(x): Passes the input through the sequential layers and returns
                    the output.
    """

    def __init__(self, inputs, neurons, outputs):
        super().__init__()

        input_vec = [inputs] + outputs[:-1]

        self.layers = Sequential(*[vanilla(i, n, o)
                                   for i, n, o in zip(input_vec,
                                                      neurons, outputs)])

    def forward(self, x):
        """
        Pass input through the network layers.

        Parameters
        ----------
            x (torch.Tensor): Input tensor of shape (batch_size, inputs).

        Returns
        -------
            torch.Tensor: Output tensor after passing through all layers.
        """
        y = self.layers(x)

        return y


def train(model, optimizer, loss_func, num_epochs, device, train_loader,
          val_loader):
    """
    Train a neural network model using mini-batch gradient descent.

    This function iterates over the dataset for a specified number of epochs,
    computes the loss for each batch, updates model parameters using
    backpropagation, and evaluates the model on both training and validation
    datasets at the end of each epoch.

    Parameters
    ----------
        model (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model
                                           parameters.
        loss_func (callable): The loss function used to compute the error.
        num_epochs (int): The total number of training epochs.
        device (torch.device): The device (CPU or GPU) on which to perform
                               computations.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training
                                                    dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation
                                                  dataset.

    Returns
    -------
        tuple: A tuple containing average losses per epoch:
            - train_losses (list of float): List of average training losses.
            - val_losses (list of float): List of average validation losses.
    """
    # Initialize lists to track losses.
    train_losses, val_losses = [], []

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
            loss = loss_func(target_batch, pred)

            # Calculate loss gradients
            loss.backward()
            # Update model weights using loss gradients
            optimizer.step()

        train_loss, val_loss = evaluate_model(model, loss_func, train_loader,
                                              val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss {train_losses[-1]:.3f}, \
              Val loss {val_losses[-1]:.3f}")

    return train_losses, val_losses


def evaluate_model(model, loss_func, train_loader, val_loader, device):
    """
    Evaluate a model on both training and validation datasets.

    This function temporarily sets the model to evaluation mode, computes
    the average loss over the training and validation datasets using the given
    loss function, and then restores the model to training mode.

    Parameters
    ----------
        model (torch.nn.Module): The neural network model to be evaluated.
        loss_func (callable): The loss function used to compute the loss.
        train_loader (torch.utils.data.DataLoader): DataLoader for training.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation.
        device (torch.device): The device (CPU or GPU) on which to perform
                               computations.

    Returns
    -------
        tuple: A tuple containing:
            - train_loss (float): The average loss on the training dataset.
            - val_loss (float): The average loss on the validation dataset.
    """
    model.eval()

    with no_grad():
        train_loss = calc_loss_loader(train_loader, model, loss_func, device)
        val_loss = calc_loss_loader(val_loader, model, loss_func, device)

    model.train()

    return train_loss, val_loss
