# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:07:24 2025.

@author: kayol
"""

from torch import nn, tensor, randn, ones, zeros, cfloat, exp, sqrt, mean, var
from torch import max as max_torch


class vanilla(nn.Module):
    """
    A vanilla PT-RBF layer.

    This layer performs a complex Gaussian basis transformation followed by a
    linear projection.

    Attributes
    ----------
    G : torch.nn.Parameter
        A complex-valued weight matrix of shape (neurons, inputs) for the basis
        transformation (matrix of center vectors).
    s : torch.nn.Parameter
        A complex-valued scaling parameter of shape (neurons, 1) controlling
        the spread of the basis functions (vector of center variance).
    W : torch.nn.Parameter
        A complex-valued weight matrix of shape (outputs, neurons) for the
        final transformation (matrix of synaptic weights).
    b : torch.nn.Parameter
        A complex-valued bias vector of shape (outputs, 1).

    Parameters
    ----------
    inputs : int
        Number of input features.
    neurons : int
        Number of neurons in the hidden layer.
    outputs : int
        Number of output features.
    """

    def __init__(self, inputs, neurons, outputs, eps=1e-5):
        """
        Initialize the Vanilla layer with complex-valued parameters.

        The optimal initialization follows the paper:
        J. A. Soares, K. S. Mayer and D. S. Arantes, "On the Parameter
        Selection of Phase-transmittance Radial Basis Function Neural Networks
        for Communication Systems," 2024 IEEE International Conference on
        Machine Learning for Communication and Networking (ICMLCN), Stockholm,
        Sweden, 2024, pp. 530-536, doi: 10.1109/ICMLCN59089.2024.10624891.

        Parameters
        ----------
        inputs : int
            Number of input features.
        neurons : int
            Number of neurons in the hidden layer.
        outputs : int
            Number of output features.
        """
        super().__init__()

        # Initialize the variance limitaion of the Gaussian Neurons
        self.eps = tensor(eps)

        # Create the sinaptic weight matrix with zero mean and unitary variance
        W = randn(neurons, outputs, dtype=cfloat)
        W = (W-mean(W, dim=0, keepdim=True))/sqrt(var(W, dim=0, keepdim=True))

        # Create the center vector matrix with zero mean and unitary variance
        G = randn(inputs, neurons, dtype=cfloat)
        G = (G-mean(G, dim=0, keepdim=True))/sqrt(var(G, dim=0, keepdim=True))

        self.G = nn.Parameter(G/sqrt(tensor([inputs])))
        self.s = nn.Parameter(ones(1, neurons) + 1j * ones(1, neurons))
        self.W = nn.Parameter(W*sqrt(5 * exp(tensor([2])) * inputs /
                                     (12 * neurons * outputs)))
        self.b = nn.Parameter(zeros(1, outputs, dtype=cfloat))

    def forward(self, x):
        """
        Perform the forward pass of the layer.

        The input is transformed using a split-complex Gaussian function, and
        then linearly projected.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, inputs, 1), expected to be
            complex-valued.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, outputs, 1), containing
            complex-valued predictions.
        """
        v_real = ((x.real - self.G.real) ** 2).sum(dim=1, keepdim=True)
        v_imag = ((x.imag - self.G.imag) ** 2).sum(dim=1, keepdim=True)

        phi = exp(- v_real / max_torch(self.s.real, self.eps)) +\
            1j * exp(- v_imag / max_torch(self.s.imag, self.eps))

        y = phi @ self.W + self.b

        return y
