#Packages needed for the neural network
import numpy as np
import torch
import torch.nn as nn            # Base class used to develop all neural network models
import torch.optim as optim      # Module of Adam optimizer
from torch.utils.data import DataLoader #Easy and organized data loading to the ML model
from torch.utils.data import Dataset  #Nice loadable dataset creation


#Amplitude domain the data was generated over - used to normalize amplitude into [-1,1]
AMP_MIN, AMP_MAX = 0.0, 5.9


def normalize_amplitude(amp):
    return 2*(amp - AMP_MIN)/(AMP_MAX - AMP_MIN) - 1


#Coustm Data for the fortmat of DataLoader (getitem is the important instruction)
class CustomData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    #Gives size of the data set
    def __len__(self):
        return len(self.data)

    #Given an idx, gives data and label (Needed for DataLoader)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]


#Create the Neural Network
class NeuralNetwrok(nn.Module):
    def __init__(self):
        #Initialize nn.Module
        super().__init__()
        #Just a precaution, here already flattened
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_stack(X)
        return logits


#Define a function for the training of the model in each epoch
def train_one_epoch(dataloader, model, loss_function, optimizer):
    #Set the model in training mode
    model.train()
    #Set variable to quantify the loss
    running_loss = 0
    num_batches = len(dataloader)

    #For every batch
    for X, y in dataloader:
        # Forward pass
        pred = model(X)
        loss = loss_function(pred, y)

        # Backpropagation (Update weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()

    return running_loss / num_batches


def test_model(dataloader, model, loss_function):
    #Set model in evaluation mode
    model.eval()
    running_loss = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            #Only forward propagation to check for loss
            pred = model(X)
            loss_t = loss_function(pred, y)
            running_loss += loss_t.item()

    return running_loss / num_batches


def prepare_data(data, seed = 15):
    """Normalize, transform to tensor and split in train/test"""
    #Reshaped to (n,1) so each sample already has the shape nn.Linear(1,...) expects
    X = torch.from_numpy(normalize_amplitude(data[:, 0]).reshape(-1, 1)).float()
    y = torch.from_numpy(data[:, 1].reshape(-1, 1)).float()

    n = len(data)
    n_train = int(0.8*n)
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(n)

    train_dataset = CustomData(X[indexes[:n_train]], y[indexes[:n_train]])
    test_dataset = CustomData(X[indexes[n_train:]], y[indexes[n_train:]])

    return train_dataset, test_dataset


def train_model(train_dataset, test_dataset, batch_size = 32, learning_rate = 0.001, epoch = 60):
    """Train the network and return it together with the loss curves"""
    #Loss function is Mean Square Error as it says in the paper
    loss_f = nn.MSELoss()

    dataloader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dataloader_test = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    #Initialize the neural network
    model = NeuralNetwrok()
    #The omptimizer chosen is Adam
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_curve, test_curve = [], []
    for e in range(epoch):
        #Training phase
        run_loss_tr = train_one_epoch(dataloader_train, model, loss_f, optimizer)
        #Test phase
        run_loss_test = test_model(dataloader_test, model, loss_f)
        train_curve.append(run_loss_tr)
        test_curve.append(run_loss_test)

    return model, train_curve, test_curve


def predict(model, amplitudes, chunk_size = 100000):
    """Predicted fidelities for an array of amplitudes"""
    amplitudes_t = torch.from_numpy(normalize_amplitude(amplitudes).reshape(-1, 1)).float()

    model.eval()
    fidelities = []
    with torch.no_grad():
        for chunk in torch.split(amplitudes_t, chunk_size):
            fidelities.append(model(chunk))

    return torch.cat(fidelities).numpy().squeeze()


def find_best_amplitude(model, epsilon = 1e-3, amp_max = 5.9):
    """Two stage search over the trained network"""
    #First stage, coarse grid
    amplitudes_coarse = np.linspace(0, amp_max, int(amp_max/epsilon))
    fidelities_coarse = predict(model, amplitudes_coarse)
    max_amp_coarse = amplitudes_coarse[fidelities_coarse.argmax()]

    #Second stage, fine grid around the best amplitude
    amplitudes_fine = np.linspace(max_amp_coarse - epsilon, max_amp_coarse + epsilon, int(2*epsilon/1e-6))
    fidelities_fine = predict(model, amplitudes_fine)
    max_amp_fine = amplitudes_fine[fidelities_fine.argmax()]

    return max_amp_fine, (amplitudes_coarse, fidelities_coarse), (amplitudes_fine, fidelities_fine)
