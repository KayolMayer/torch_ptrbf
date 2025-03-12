"""
Created on Wed Mar 12 16:27:24 2025.

@author: kayol
"""

from torch import abs


def mse_loss(output, target):
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
    return (abs(output - target)**2).sum(dim=1).mean(dim=0)
