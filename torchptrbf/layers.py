# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 15:07:24 2025.

@author: kayol
"""

from torch import nn, tensor, randn, ones, zeros, cfloat, exp, sqrt, mean, var


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

    def __init__(self, inputs, neurons, outputs):
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

        # Create the sinaptic weight matrix with zero mean and unitary variance
        W = randn(outputs, neurons, dtype=cfloat)
        W = (W - mean(W, dim=-1, keepdim=True)) / sqrt(var(W, dim=-1,
                                                           keepdim=True))

        # Create the center vector matrix with zero mean and unitary variance
        G = randn(neurons, inputs, dtype=cfloat)
        G = (G - mean(G, dim=-1, keepdim=True)) / sqrt(var(G, dim=-1,
                                                           keepdim=True))

        self.G = nn.Parameter(G*sqrt(tensor([inputs])))
        self.s = nn.Parameter(ones(neurons, 1) + 1j * ones(neurons, 1))
        self.W = nn.Parameter(W*sqrt(5 * exp(tensor([2])) * inputs /
                                     (12 * neurons * outputs)))
        self.b = nn.Parameter(zeros(outputs, 1, dtype=cfloat))

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
        v_real = ((self.G.real * x.real.transpose(1, 2)) ** 2).\
            sum(dim=2, keepdim=True)
        v_imag = ((self.G.imag * x.imag.transpose(1, 2)) ** 2).\
            sum(dim=2, keepdim=True)

        phi = exp(- v_real / self.s.real) + 1j * exp(- v_imag / self.s.imag)

        y = self.W @ phi + self.b

        return y
