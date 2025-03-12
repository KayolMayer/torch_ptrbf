"""
Created on Wed Mar 12 17:38:12 2025.

@author: kayol
"""

import numpy as np
from torch import from_numpy, cfloat, mean, var, sqrt, tensor
from torch.utils.data import Dataset, DataLoader


class beamforming_dataset(Dataset):

    def __init__(self, len_data=1e4):
        self.input_ids = []
        self.target_ids = []

        input_data, output_data = dataset_beamforming_gen(len_data=len_data)

        input_data = (input_data - mean(input_data, dim=0, keepdim=True)) / \
        sqrt(var(input_data, dim=-1, keepdim=True)) / sqrt(tensor(input_data.shape[1]))

        output_data = (output_data - mean(output_data, dim=0, keepdim=True)) / \
        sqrt(var(output_data, dim=-1, keepdim=True)) / sqrt(tensor(output_data.shape[1]))

        self.input_ids = [input_data[ii].reshape((-1,1)) for ii in range(int(len_data))]
        self.target_ids = [output_data[ii].reshape((-1,1)) for ii in range(int(len_data))]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def beamforming_dataset_loader(len_data=1e4, batch_size=1, shuffle=True,
                               drop_last=True, num_workers=0):

    # Create dataset
    dataset = beamforming_dataset(len_data=len_data)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def dataset_beamforming_gen(len_data=1e4):
    """
    Create a dataset for beamforming.

    Parameters
    ----------
        modulation (list): List of modulation types.
        Mmod (list): List of modulation orders.
        f (float): Frequency of the signal.
        phi (float): Angle in degrees for the transmitter.
        theta (float): Angle in degrees for the elevation.
        desired (array): Array indicating if each output is desired or not.
        lenData (int): Length of the dataset.
        SINRdB (float): Signal-to-Interference-plus-Noise Ratio in dB.
        SNRdBs (float): Signal-to-Noise Ratio in dB for sources.
        SNRdBi (float): Signal-to-Noise Ratio in dB for interferences.

    Returns
    -------
        tuple: Tuple containing the input and output datasets.
    """
    f = 850e6
    SINRdB = 20
    SNRdBs = 25
    SNRdBi = 20
    phi = [1, 60, 90, 120, 160, 200, 240, 260, 280, 300, 330]
    theta = [90] * 11
    desired = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    modulation = ["QAM", "WGN", "QAM", "PSK", "QAM", "WGN", "QAM", "WGN",
                  "QAM", "PSK", "PSK"]
    Mmod = [4, 0, 64, 8, 256, 0, 16, 0, 64, 16, 8]

    lenData = int(len_data)

    # Linear SNR of sources
    SNRs = 10**(SNRdBs/10)
    # Linear SNR of interferences
    SNRi = 10**(SNRdBi/10)
    # Linear SINR
    SINR = 10**(SINRdB/10)

    # Number of sources
    Ns = np.sum(desired)
    # Number of interferences
    Ni = len(desired) - Ns

    # Normalization factor to the desired SINR
    iota = (Ni/Ns) * (1/SNRi + 1) / (1/SINR - 1/SNRs)

    # Standard deviations for AWGN noises of sources and interferences
    StdDevS = np.sqrt((1/SNRs)/2)
    StdDevI = np.sqrt((1/(iota*SNRi))/2)

    # Create StdDevVect for each modulation type
    StdDevVect = np.array([np.where(desired == 1, StdDevS, StdDevI)]).T

    # Speed of light in vacuum [m/s]
    c = 299792458
    # Wavelength
    lambda_val = c / f
    # Propagation constant
    beta = 2 * np.pi / lambda_val
    # Dipoles length
    L = 0.5 * lambda_val
    # Dipoles spacing
    s = 0.25 * lambda_val
    # Dipoles coordinates
    coord = np.array([
        [s, 0, 0],
        [s * np.cos(np.deg2rad(60)), s * np.sin(np.deg2rad(60)), 0],
        [-s * np.cos(np.deg2rad(60)), s * np.sin(np.deg2rad(60)), 0],
        [-s, 0, 0],
        [-s * np.cos(np.deg2rad(60)), -s * np.sin(np.deg2rad(60)), 0],
        [s * np.cos(np.deg2rad(60)), -s * np.sin(np.deg2rad(60)), 0]
    ]).T

    # Matrix of self and mutual impedances
    Z = np.array([
        [78.424+45.545j, 46.9401-32.6392j, -0.791-41.3825j, -14.4422-34.4374j,
         -0.791-41.3825j, 46.9401-32.6392j],
        [46.9401-32.6392j, 78.424+45.545j, 46.9401-32.6392j, -0.791-41.3825j,
         -14.4422-34.4374j, -0.791-41.3825j],
        [-0.791-41.3825j, 46.9401-32.6392j, 78.424+45.545j, 46.9401-32.6392j,
         -0.791-41.3825j, -14.4422-34.4374j],
        [-14.4422-34.4374j, -0.791-41.3825j, 46.9401-32.6392j, 78.424+45.545j,
         46.9401-32.6392j, -0.791-41.3825j],
        [-0.791-41.3825j, -14.4422-34.4374j, -0.791-41.3825j, 46.9401-32.6392j,
         78.424+45.545j, 46.9401-32.6392j],
        [46.9401-32.6392j, -0.791-41.3825j, -14.4422-34.4374j, -0.791-41.3825j,
         46.9401-32.6392j, 78.424+45.545j]
    ])

    # Load impedance
    ZT = 50  # [ohms]

    # Dipoles self impedance
    ZA = Z[0, 0]

    # Coupling matrix
    C = (ZT + ZA) * np.linalg.inv((Z + ZT * np.eye(6)))

    # Matrix of relative intensity of Etheta
    Xm = np.diag((lambda_val / (np.pi * np.sin(beta * L / 2))) *
                 ((np.cos(L * np.pi * np.cos(np.deg2rad(theta)) / lambda_val) -
                   np.cos(np.pi * L / lambda_val)) /
                  np.sin(np.deg2rad(theta))))

    # Matrix of Tx angular positions
    Omega = np.array([np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)),
                      np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)),
                      np.cos(np.deg2rad(theta))]).T

    Pi = np.exp(-2j * np.pi * np.dot(Omega, coord) / lambda_val)

    # Steering vectors
    psi = np.dot(Pi.T, Xm)

    # Create symbols of sources and interferences
    SetOut = np.zeros((len(modulation), lenData), dtype=complex)

    for ii, mod_type in enumerate(modulation):
        # QAM symbols
        if mod_type == "QAM":
            if Mmod[ii] != 0:
                SetOut[ii] = (((np.random.randint(np.sqrt(Mmod[ii]),
                                                  size=lenData)) * 2) -
                              (np.sqrt(Mmod[ii]) - 1)) + \
                            1j * (((np.random.randint(np.sqrt(Mmod[ii]),
                                                      size=lenData)) * 2) -
                                  (np.sqrt(Mmod[ii]) - 1))
            else:
                SetOut[ii] = np.random.randn(1, lenData) + \
                                            1j * np.random.randn(1, lenData)
        # PSK symbols
        elif mod_type == "PSK":
            if Mmod[ii] != 0:
                pskAng = np.random.randint(Mmod[ii], size=lenData) * 2 \
                    * np.pi / Mmod[ii]
                SetOut[ii] = np.cos(pskAng) + 1j * np.sin(pskAng)
            else:
                SetOut[ii] = np.random.randn(1, lenData) + 1j \
                    * np.random.randn(1, lenData)
        # WGN noise
        else:
            SetOut[ii] = np.random.randn(1, lenData) + \
                1j * np.random.randn(1, lenData)

    # Compute the source and interference powers for normalization
    P = np.sum((np.abs(SetOut - np.mean(SetOut, axis=1)[:, np.newaxis])**2),
               axis=1) / lenData

    # Normalize the powers to 1
    SetOut = SetOut / np.sqrt(P[:, np.newaxis])

    # Interference normalizations to the desired SINR
    SetOut[np.where(desired == 0)[0], :] = \
        SetOut[np.where(desired == 0)[0], :] / np.sqrt(iota)

    # Create the data that impinged on the beamforming
    SetIn = np.dot(C, psi).dot(SetOut + StdDevVect *
                               (np.random.randn(len(modulation), lenData) +
                                1j * np.random.randn(len(modulation),
                                                     lenData)))

    # Compute the signal powers in each RX dipole
    P = np.sum((np.abs(SetIn - np.mean(SetIn, axis=1)[:, np.newaxis])**2),
               axis=1) / lenData

    # Normalize the signals in each dipole to have a unitary power
    SetIn = SetIn / np.sqrt(P[:, np.newaxis])

    # Add nonlinearities
    SetIn = SetIn - 0.1 * SetIn**3 - 0.05 * SetIn**5

    # Select desired outputs
    indices_true = np.where(desired)
    SetOut = SetOut[indices_true]

    return from_numpy(SetIn.T).to(cfloat), from_numpy(SetOut.T).to(cfloat)
