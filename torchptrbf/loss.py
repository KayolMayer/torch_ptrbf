"""
Created on Wed Mar 12 16:27:24 2025.

@author: kayol
"""

from torch import abs


def mse_loss(target, output):
    """
    Compute the Mean Squared Error (MSE) loss between the output and target.

    The MSE loss is defined as the mean of the squared absolute differences
    between the predicted and target values.

    Parameters
    ----------
    output : torch.Tensor (scalar)
        The predicted values, expected to be a complex-valued tensor.
    target : torch.Tensor
        The ground truth values, expected to be a complex-valued tensor of the
        same shape as `output`.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the computed MSE loss.
    """
    return 0.5 * (abs(target - output) ** 2).sum(dim=2).mean(dim=0)


def calc_loss_loader(data_loader, model, device):
    """
    Compute the average cross-entropy loss over multiple batches from a loader.

    This function iterates through a specified number of batches from the given
    data loader, computes the loss for each batch and returns the average loss.

    Parameters
    ----------
        data_loader (torch.utils.data.DataLoader): The data loader providing
                                                   batches of input and target
                                                   tensors.
        model (torch.nn.Module): The NN model.
        device (torch.device): The device (CPU or GPU) on which to perform
                               computations.
    Returns
    -------
        float: The average loss.
               Returns NaN if the data loader is empty.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    else:
        # Reduce the number of batches to match the total number of batches in
        # the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):

        # Pass data to device
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # Compute prediction error
        pred = model(input_batch)

        loss = mse_loss(target_batch, pred)
        total_loss += loss.item()

    return total_loss / num_batches
