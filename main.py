"""
Created on Wed Mar 12 16:22:22 2025.

@author: kayol
"""
import torch
import torch.nn as nn

from torchptrbf.layers   import vanilla
from torchptrbf.loss     import mse_loss
from torchptrbf.datasets import beamforming_dataset_loader

import matplotlib.pyplot as plt


class ptrbf_nn(nn.Module):

    def __init__(self, inputs, neurons, outputs):
        super().__init__()

        input_vec = [inputs] + outputs[:-1]

        self.layers = nn.Sequential(*[vanilla(i, n, o)
                                      for i, n, o in zip(input_vec,
                                                         neurons, outputs)])

    def forward(self, x):

        y = self.layers(x)

        return y

def train(dataset, model, loss_fn, optimizer, device):

    size = len(dataset)
    model.train()
    for batch, (X, y) in enumerate(dataset):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataset, model, loss_fn, device):
    size = len(dataset)
    num_batches = dataset.batch_size
    model.eval()
    test_loss, error = 0, 0
    with torch.no_grad():
        for X, y in dataset:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            error += sum(sum(abs(y-pred))).item()

            #plt.scatter(pred[:,2,0].real,pred[:,2,0].imag)
    test_loss /= num_batches
    error /= (size*num_batches)
    print(f"Test Error: {error:>8f}, Avg loss: {test_loss:>8f} \n")


batch_size = 10
len_data   = 1e5
epochs     = 100

# Create data loaders
train_dataloader = beamforming_dataset_loader(len_data=len_data, batch_size=batch_size, shuffle=True)
test_dataloader  = beamforming_dataset_loader(len_data=len_data/100, batch_size=100)

# Get device for training
#device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
print(f"Using {device} device")


inputs  = 6
outputs = [50, 3]
neurons = [50, 50]
ptrbf = ptrbf_nn(inputs, neurons, outputs).to(device)
print(ptrbf)

loss_fn = mse_loss
optimizer = torch.optim.Adam(ptrbf.parameters(), lr=1e-4)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, ptrbf, loss_fn, optimizer, device)
    test(test_dataloader, ptrbf, loss_fn, device)
print("Done!")























# inputs  = 5
# outputs = [10, 10, 10]
# neurons = [5, 6, 7]

# ptrbf = ptrbf_nn(inputs, neurons, outputs)

# x      = torch.randn(1,inputs, 1)+torch.randn(1,inputs, 1)*1j
# target = torch.ones(1, outputs[-1], 1)+torch.ones(1, outputs[-1], 1)*1j

# optimizer = torch.optim.Adam(ptrbf.parameters(), lr=1e-3)

# for _ in range(200):

#     # Forward pass
#     output = ptrbf(x)

#     # Calculate loss based on how close the target
#     # and output are
#     loss      = mse_loss(output, target)

#     # Backward pass to calculate the gradients
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()


# # In[173]:


# print(mse_loss(output, target))
