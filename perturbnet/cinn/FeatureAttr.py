import sys
import os
import time
import random

import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from captum.attr import IntegratedGradients
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps




class BinaryCellStatesClass(nn.Module):
    """ Class to predict the label for two latent spaces """
    def __init__(self, z_dim = 10, hidden_dim = 32, prob_drop = 0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear_1 = nn.Linear(z_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p = prob_drop)
        self.train_loss = []
        self.valid_loss = []
        self.epoch = []

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.sigmoid(self.linear_2(x))
        return x
    
    def train_model(self, dataset, num_epochs=20, batch_size=32, learning_rate=0.001, val_split=0.1):
        # Split the dataset into training and validation sets
        train_size = int((1 - val_split) * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for epoch in tqdm(range(num_epochs)):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs.to(device)).squeeze()
                loss = criterion(outputs.to(device), labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Calculate average training loss
            train_loss = running_loss / len(train_loader)
            self.train_loss.append(train_loss)
            
            # Validate the model
            valid_loss = self.validate(valid_loader, criterion)
            self.valid_loss.append(valid_loss)
            self.epoch.append(epoch)
            
    def validate(self, dataloader, criterion):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs.to(device)).squeeze()
                loss = criterion(outputs.to(device), labels.to(device))
                val_loss += loss.item()
        return val_loss / len(dataloader)
    
    def plot_losses(self):
        epochs = self.epoch
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_loss, 'b', label='Training loss')
        plt.plot(epochs, self.valid_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def save(self, dir_path: str, overwrite: bool = False):
        if not os.path.exists(dir_path) or overwrite: 
            os.makedirs(dir_path, exist_ok = overwrite)
        else:
            raise ValueError(
                "{} already exists, Please provide an unexisting director for saving.".format(dir_path))
        model_save_path = os.path.join(dir_path, "model_params.pt")

        torch.save(self.state_dict(), model_save_path)

        np.save(os.path.join(dir_path, "training_epoch.npy"), self.epoch)
        np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
        np.save(os.path.join(dir_path, "test_loss.npy"), self.valid_loss)
        
    def load(self, dir_path: str, use_cuda: bool = False, save_use_cuda: bool = False):
        use_cuda = use_cuda and torch.cuda.is_available()
        device = torch.device('cpu') if use_cuda is False else torch.device('cuda')

        model_save_path = os.path.join(dir_path, "model_params.pt")
        

        if use_cuda and save_use_cuda:
            self.load_state_dict(torch.load(model_save_path))
            self.to(device)
        elif use_cuda and save_use_cuda is False:
            self.load_state_dict(torch.load(model_save_path, map_location = "cuda:0"))
            self.to(device)
        else:
            self.load_state_dict(torch.load(model_save_path, map_location = device))


class FlowResizeLabelClass(nn.Module):
    """Class to generate cellular representations via PerturbNet from perturbation onehot encodings"""
    def __init__(self,
                 model,
                 model_g,
                 model_class,
                 std_model,
                 zDim,
                 yDim,
                 n_seq,
                 n_vol,
                 device):
        super().__init__()
        self.model = model
        self.model_g = model_g
        self.model_class = model_class
        self.std_model = std_model
        self.zDim = zDim
        self.yDim = yDim
        self.n_seq = n_seq
        self.n_vol = n_vol
        self.device = device

    def generte_zprime(self,  x, c, cprime):
        zz, _ = self.model.flow(x, c)
        zprime = self.model.flow.reverse(zz, cprime)
        return zprime

    def forward(self, input_data, batch_size = 50):
        latent = input_data[:, :self.zDim]

        condition = input_data[:, self.zDim:(self.zDim + self.yDim)]
        trt_onehot = input_data[:, (self.zDim + self.yDim):]
        trt_onehot = trt_onehot.view(trt_onehot.size(0), self.n_seq, self.n_vol)

        _, _, _, embdata_torch_sub = self.model_g(trt_onehot.float())
        condition_new = self.std_model.standardize_z_torch(embdata_torch_sub)

        trans_z = self.generte_zprime(latent.float().to(self.device).unsqueeze(-1).unsqueeze(-1),
                                             condition.float().to(self.device),
                                             condition_new.float().to(self.device)
            ).squeeze(-1).squeeze(-1)#.cpu().detach().numpy()

        trans_z_class = self.model_class(trans_z)

        return trans_z_class


class FlowResizeYLabelClass(nn.Module):
    """Class to generate cellular representations via PerturbNet from perturbation representations"""
    def __init__(self,
                 model,
                 model_g,
                 model_class,
                 std_model,
                 zDim,
                 yDim,
                 n_seq,
                 n_vol,
                 device):
        super().__init__()
        self.model = model
        self.model_g = model_g
        self.model_class = model_class
        self.std_model = std_model
        self.zDim = zDim
        self.yDim = yDim
        self.n_seq = n_seq
        self.n_vol = n_vol
        self.device = device

    def generte_zprime(self,  x, c, cprime):
        zz, _ = self.model.flow(x, c)
        zprime = self.model.flow.reverse(zz, cprime)
        return zprime

    def forward(self, input_data, batch_size = 50):
        latent = input_data[:, :self.zDim]

        condition = input_data[:, self.zDim:(self.zDim + self.yDim)]
        trt_onehot = input_data[:, (self.zDim + self.yDim):]

        #trt_onehot = trt_onehot.view(trt_onehot.size(0), self.n_seq, self.n_vol)
        #_, _, _, embdata_torch_sub = self.model_g(trt_onehot.float())
        #condition_new = self.std_model.standardize_z_torch(embdata_torch_sub)

        condition_new = trt_onehot.float()

        trans_z = self.generte_zprime(latent.float().to(self.device).unsqueeze(-1).unsqueeze(-1),
                                             condition.float().to(self.device),
                                             condition_new.float().to(self.device)
            ).squeeze(-1).squeeze(-1)#.cpu().detach().numpy()

        trans_z_class = self.model_class(trans_z)

        return trans_z_class


def ig_b_score_compute(ig, input_data, baseline_null, target, batch_size = 32, ifPlot = False, plot_save_file = '.'):
    """
    Integrated gradients attributions of onehot encodings of perturbations
    """
    n = input_data.shape[0]
    n_batches = n // batch_size
    if n_batches * batch_size < n:
        n_batches += 1

    attr_ig = None
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)

        attributions, delta_ig = ig.attribute(input_data[start:end],
                                              baseline_null[start:end],
                                              target = target,
                                              return_convergence_delta = True)
        attributions = attributions[:, (10 + 196):].view(attributions.size(0), 120, 35)
        if attr_ig is None:
            attr_ig = attributions.cpu().detach().numpy()
        else:
            attr_ig = np.concatenate((attr_ig, attributions.cpu().detach().numpy()), axis = 0)

    if ifPlot:
        newfig = plt.figure(figsize = (20, 10))
        plt.imshow(attr_ig.mean(0).T, cmap = 'viridis')
        plt.colorbar()
        newfig.savefig(plot_save_file)

    return attr_ig.mean(0)


def ig_y_score_compute(ig, input_data, baseline_null, target, batch_size = 32, ifPlot = False, plot_save_file = '.'):
    """
    Integrated gradients attributions of representations of perturbations
    """
    n = input_data.shape[0]
    n_batches = n // batch_size
    if n_batches * batch_size < n:
        n_batches += 1

    attr_ig = None
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)

        attributions, delta_ig = ig.attribute(input_data[start:end],
                                              baseline_null[start:end],
                                              target = target,
                                              return_convergence_delta = True)
        attributions = attributions[:, (10 + 196):]#.view(attributions.size(0), 120, 35)
        if attr_ig is None:
            attr_ig = attributions.cpu().detach().numpy()
        else:
            attr_ig = np.concatenate((attr_ig, attributions.cpu().detach().numpy()), axis = 0)

    if ifPlot:
        newfig = plt.figure(figsize = (20, 10))
        mean_attr_ig = attr_ig.mean(axis=0)
        abs_mean_attr_ig = np.abs(mean_attr_ig)
        # Get the indices of the top 10 features
        top_10_indices = np.argsort(abs_mean_attr_ig)[-10:]

        # Extract the values of the top 10 features
        top_10_values = mean_attr_ig[top_10_indices]
        plt.bar(range(len(top_10_values)), top_10_values, tick_label=top_10_indices)
        plt.xlabel('Feature Index')
        newfig.savefig(plot_save_file)

    return attr_ig.mean(0)


class CellDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    
    
def plot_molecule_attribtuion_score(smile, onehot_matrix, attribution_scores, cluster = None, chem_id = None, save_path = None, dpi = 600,
                                   contourLines = 10 ):
    # Define your molecule and atom scores
    molecule = Chem.MolFromSmiles(smile)
    atom_indices = np.argmax(onehot_matrix, axis=1)
    atom_indices = atom_indices[:len(smile)]
    #print(len(atom_indices))
    # Aggregate attribution scores per atom
    atom_scores = np.zeros(atom_indices.shape[0])
    #print(atom_scores.shape)
    #print(len(smiles))
    for i, idx in enumerate(atom_indices):
        atom_scores[i] += attribution_scores[i, idx]
    max_abs = np.max(np.abs(atom_scores))


    # Generate the similarity map and get the figure
    fig = SimilarityMaps.GetSimilarityMapFromWeights(molecule, atom_scores, contourLines=contourLines)

    # Set the colormap to center at zero
    vmin = -max(abs(atom_scores))
    vmax = max(abs(atom_scores))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Create the ScalarMappable and color bar
    sm = plt.cm.ScalarMappable(cmap='PiYG', norm=norm)  # 'seismic' colormap is good for centering at zero
    sm.set_array([])

    # Add the color bar to the figure
    cbar = fig.colorbar(sm)
    #cbar.set_label('Atom Score')

    # Display the figure
    if cluster and chem_id:
        plt.title(chem_id + ": " + "Cluster" + cluster)
    if save_path:
        plt.savefig(save_path, dpi = dpi, bbox_inches='tight')
    plt.show()

