"""
Created on Wed Mar 12 16:22:22 2025.

@author: kayol
"""
from torch import no_grad, optim, manual_seed
from torch.cuda import is_available

from numpy.random import seed

from torchptrbf.loss import mse_loss
from torchptrbf.datasets import beamforming_dataset_loader
from torchptrbf.architectures import ptrbf_vanilla, train
from torchptrbf.plots import plot_losses, plot_consts


def inference(inference_loader, model, device):
    """
    Perform inference using a trained model on the provided dataset.

    This function iterates through the `inference_loader`, passes each input
    batch through the model, and extracts specific predictions for QAM-4,
    PSK-8, and QAM-16 modulation schemes.

    Parameters
    ----------
        inference_loader (torch.utils.data.DataLoader): DataLoader providing
                                                        batches of inputs.
        model (torch.nn.Module): The neural network model used for inference.
        device (torch.device): The device (CPU or GPU) on which to perform
                               computations.

    Returns
    -------
        tuple: A tuple containing three lists:
            - qam4 (list): Predictions for QAM-4 modulation.
            - psk8 (list): Predictions for PSK-8 modulation.
            - qam16 (list): Predictions for QAM-16 modulation.
    """
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


# PT-RBF parameters
ptrbf_arc = {
    'inputs': 6,
    'outputs': [50, 3],
    'neurons': [50, 50],
    'batch': 1,
    'lr': 3e-3,
    'epochs': 2,
    'len_train': 1e4,
    'len_val': 1e3,
    'len_inf': 1e4,
    'random_seed': 123
    }

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

ptrbf = ptrbf_vanilla(ptrbf_arc['inputs'], ptrbf_arc['neurons'],
                      ptrbf_arc['outputs']).to(device)

optimizer = optim.SGD(ptrbf.parameters(), lr=ptrbf_arc['lr'])

# Model training
train_losses, val_losses = train(
    ptrbf, optimizer, mse_loss, ptrbf_arc['epochs'], device,
    train_loader, val_loader)

# Plot the training and validation losses.
plot_losses(ptrbf_arc['epochs'], train_losses, val_losses, 'losses')

# Model inference
qam4, psk8, qam16 = inference(inf_loader, ptrbf, device)

plot_consts(qam4, "QAM-4")
plot_consts(psk8, "PSK-8")
plot_consts(qam16, "QAM-16")
