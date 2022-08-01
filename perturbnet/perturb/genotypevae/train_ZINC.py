#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from genotypeVAE import *
import torch

if __name__ == "__main__":

    path_save = ""
    if not os.path.exists(path_save):
        os.makedirs(path_save, exist_ok = True)

    path_data = ""
    path_lincs_onehot = ""
    path_geno_onehot = ""

    ########################
    ## Data preparation
    ########################
    # Onehot data
    data_geno_onehot = load_npz(path_geno_onehot).toarray()

    random_state = np.random.RandomState(seed = 123)
    permutation = random_state.permutation(len(data_geno_onehot))
    n_train = int(len(data_geno_onehot) * 0.8)
    n_test = len(data_geno_onehot) - n_train
    batch_size = 128

    data_train, data_test = data_geno_onehot[permutation[:n_train]], data_geno_onehot[permutation[n_train:]]

    data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
    data_test = torch.utils.data.TensorDataset(torch.from_numpy(data_test))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, shuffle = True)

    #################
    # Model
    ##################
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GenotypeVAE_Customize(x_dim = 15988, z_dim = 196).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    epochs = 300

    def train(epoch):

        # train
        model.train()
        train_loss = 0
        n_train = 0

        data_second = None
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device, dtype = torch.float)

            if data_second is not None:

                if data.shape[0] == data_second.shape[0]:
                    data_use = torch.logical_or(data, data_second).to(device, dtype = torch.float)
                else:
                    n_smaller = min(data.shape[0], data_second.shape[0])
                    data_use = torch.logical_or(data[:n_smaller], data_second[:n_smaller]).to(device, dtype = torch.float)

                # operations
                optimizer.zero_grad()
                output, mean, logvar, _ = model(data_use)

                # loss
                loss = vae_loss(output, data_use, mean, logvar)
                loss.backward()
                train_loss += loss.item() #* data_use.shape[0]
                optimizer.step()

                n_train += data_use.shape[0]

                data_second = None

            else:
                check = np.random.sample(1)[0]

                if check > 0.5:
                    data_second = data.detach().clone()
                    continue

                # operations
                optimizer.zero_grad()
                output, mean, logvar, _ = model(data)
                loss = vae_loss(output, data, mean, logvar)
                loss.backward()
                train_loss += loss.item() #* data.shape[0]
                optimizer.step()
                n_train += data.shape[0]

        train_loss /= n_train

        # validation
        model.eval()
        test_loss = 0
        n_test = 0
        data_second = None

        with torch.no_grad():
            for batch_test_idx, data in enumerate(test_loader):
                data = data[0].to(device, dtype = torch.float)

                if data_second is not None:
                    if data.shape[0] == data_second.shape[0]:
                        data_use = torch.logical_or(data, data_second).to(device, dtype = torch.float)
                    else:
                        n_smaller = min(data.shape[0], data_second.shape[0])
                        data_use = torch.logical_or(data[:n_smaller], data_second[:n_smaller]).to(device, dtype = torch.float)

                    output, mean, logvar, _ = model(data_use)
                    tloss = vae_loss(output, data_use, mean, logvar)

                    test_loss += tloss.item() #* data_use.shape[0]
                    n_test += data_use.shape[0]

                    data_second = None

                else:
                    check = np.random.sample(1)[0]
                    if check > 0.5:
                        data_second = data.detach().clone()
                        continue

                    # operations
                    output, mean, logvar, _ = model(data)
                    tloss = vae_loss(output, data, mean, logvar)
                    test_loss += tloss.item() #* data.shape[0]
                    n_test += data.shape[0]


            test_loss /= n_test

        return train_loss, test_loss

    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss, test_loss = train(epoch)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    pd.DataFrame(train_loss_list).to_csv(os.path.join(path_save, "train_loss.csv"))
    pd.DataFrame(test_loss_list).to_csv(os.path.join(path_save, "test_loss.csv"))
    torch.save(model.state_dict(), os.path.join(path_save, "model_params.pt"))
