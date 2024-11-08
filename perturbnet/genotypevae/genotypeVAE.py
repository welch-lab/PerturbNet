#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
	"""
	ELBO loss of VAE
	"""
	xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average = False)
	kl_loss = -0.5 * torch.sum(torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), 1))
	return xent_loss + kl_loss

class ConcatDatasetWithIndices(torch.utils.data.Dataset):
	"""
	Module of concatenating multiple datasets in a data loader
	"""
	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, i):
		return tuple([d[i] for d in self.datasets] + [i])

	def __len__(self):
		return min(len(d) for d in self.datasets)

class GenotypeVAE(nn.Module):
	"""
	GenotypeVAE module
	"""
	def __init__(self):

		super(GenotypeVAE, self).__init__()


		# encoder
		self.linear_1 = nn.Linear(15988, 512)
		self.linear_2 = nn.Linear(512, 256)
		self.linear_3_mu = nn.Linear(256, 10)
		self.linear_3_std = nn.Linear(256, 10)
		
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)

		# decoder
		self.linear_4 = nn.Linear(10, 256)
		self.linear_5 = nn.Linear(256, 512)
		self.linear_6 = nn.Linear(512, 15988)

		self.bn4 = nn.BatchNorm1d(256)
		self.bn5 = nn.BatchNorm1d(512)
		
		# activations		
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		self.leaky = nn.LeakyReLU(negative_slope = 0.2)
		self.sigmoid = nn.Sigmoid()

		self.dropout1 = nn.Dropout(p = 0.2)
		self.dropout2 = nn.Dropout(p = 0.2)
		self.dropout3 = nn.Dropout(p = 0.2)
		self.dropout4 = nn.Dropout(p = 0.2)

		
	def encode(self, x):
		# layer 1
		x = self.linear_1(x)
		x = self.bn1(x)
		x = self.leaky(x)
		x = self.dropout1(x)

		# layer 2
		x = self.linear_2(x)
		x = self.bn2(x)
		x = self.leaky(x)
		x = self.dropout2(x)

		# mean and std layers
		return self.linear_3_mu(x), self.linear_3_std(x)

	def sampling(self, z_mean, z_logvar):
		epsilon = 1e-2 * torch.randn_like(z_logvar)
		return torch.exp(0.5 * z_logvar) * epsilon + z_mean

	def decode(self, z):
		z = self.linear_4(z)
		z = self.bn4(z)
		z = self.leaky(z)
		z = self.dropout3(z)
		
		z = self.linear_5(z)
		z = self.bn5(z)
		z = self.leaky(z)
		z = self.dropout4(z)
		# z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
		# output, hn = self.gru(z)
		# out_reshape = output.contiguous().view(-1, output.size(-1))
		y0 = self.sigmoid(self.linear_6(z))
		y = y0.contiguous()
		return y

	def forward(self, x):
		z_mean, z_logvar = self.encode(x)
		self.z = self.sampling(z_mean, z_logvar)
		return self.decode(self.z), z_mean, z_logvar, self.z


class GenotypeVAE_Customize(nn.Module):
	"""
	GenotypeVAE module with customized data and latent dimensions
	"""
	def __init__(self, x_dim, z_dim):

		super().__init__()


		# encoder
		self.linear_1 = nn.Linear(x_dim, 512)
		self.linear_2 = nn.Linear(512, 256)
		self.linear_3_mu = nn.Linear(256, z_dim)
		self.linear_3_std = nn.Linear(256, z_dim)

		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)

		# decoder
		self.linear_4 = nn.Linear(z_dim, 256)
		self.linear_5 = nn.Linear(256, 512)
		self.linear_6 = nn.Linear(512, x_dim)

		self.bn4 = nn.BatchNorm1d(256)
		self.bn5 = nn.BatchNorm1d(512)

		# activations
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		self.leaky = nn.LeakyReLU(negative_slope = 0.2)
		self.sigmoid = nn.Sigmoid()

		self.dropout1 = nn.Dropout(p = 0.2)
		self.dropout2 = nn.Dropout(p = 0.2)
		self.dropout3 = nn.Dropout(p = 0.2)
		self.dropout4 = nn.Dropout(p = 0.2)

	def encode(self, x):
		# layer 1
		x = self.linear_1(x)
		print(x.shape)        
		x = self.bn1(x)
		x = self.leaky(x)
		x = self.dropout1(x)

		# layer 2
		x = self.linear_2(x)
		x = self.bn2(x)
		x = self.leaky(x)
		x = self.dropout2(x)

		# mean and std layers
		return self.linear_3_mu(x), self.linear_3_std(x)

	def sampling(self, z_mean, z_logvar):
		epsilon = 1e-2 * torch.randn_like(z_logvar)
		return torch.exp(0.5 * z_logvar) * epsilon + z_mean

	def decode(self, z):
		z = self.linear_4(z)
		z = self.bn4(z)
		z = self.leaky(z)
		z = self.dropout3(z)

		z = self.linear_5(z)
		z = self.bn5(z)
		z = self.leaky(z)
		z = self.dropout4(z)
		# z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
		# output, hn = self.gru(z)
		# out_reshape = output.contiguous().view(-1, output.size(-1))
		y0 = self.sigmoid(self.linear_6(z))
		y = y0.contiguous()
		return y

	def forward(self, x):
		z_mean, z_logvar = self.encode(x)
		self.z = self.sampling(z_mean, z_logvar)
		return self.decode(self.z), z_mean, z_logvar, self.z

class GenotypeVAETrain:
	"""
	training module to train GenotypeVAE
	"""

	def __init__(self, genovae_model, device):
		super().__init__()
		# Laplacian matrix L
		self.genovae_model = genovae_model
		self.device = device

	def train_np(self, epochs, data_vae_onehot, perturb_with_onehot, lr_vae = 0.0001,
				 batch_size_vae = 128, batch_size_vaezlz = 128, model_save_per_epochs = None, path_save = None):

		optimizer = optim.Adam(self.genovae_model.parameters(), lr = lr_vae)

		# data loaders
		random_state = np.random.RandomState(seed = 123)
		permutation = random_state.permutation(len(data_vae_onehot))
		n_train = int(len(data_vae_onehot) * 0.8)
		n_test = len(data_vae_onehot) - n_train

		data_train, data_test = torch.tensor(data_vae_onehot[permutation[:n_train]]), torch.tensor(data_vae_onehot[permutation[n_train:]])

		# ZLZ part
		n_vae = perturb_with_onehot.shape[0]
		n_vae_train = int(n_vae * 0.8)
		n_vae_test = n_vae - n_vae_train

		random_vae_state = np.random.RandomState(seed=123)
		permutation_vae = random_vae_state.permutation(n_vae)
		indices_vae_test, indices_vae_train = permutation_vae[:n_vae_test], permutation_vae[n_vae_test:]
		######

		idx_with_onehot_train = torch.tensor(np.array([indices_vae_train]).reshape(-1))
		idx_with_onehot_test = torch.tensor(np.array([indices_vae_test]).reshape(-1))

		train_prop_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(idx_with_onehot_train,
									 idx_with_onehot_train),
			batch_size = batch_size_vaezlz,
			shuffle = True)

		test_prop_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(idx_with_onehot_test,
									 idx_with_onehot_test),
			batch_size = batch_size_vaezlz,
			shuffle = True)

		# train
		train_vae_loss_list, test_vae_loss_list = [], []

		training_time = 0

		for epoch in range(1, epochs + 1):

			begin = time.time()
			self.genovae_model.train()
			train_vae_loss = 0


			for batch_idx, (_, batch_prop, _) in enumerate(train_prop_loader):
				# vae alone
				check = np.random.sample(1)[0]
				if check > 0.5:
					indices_vae_mb = random.sample(range(n_train), 2 * batch_size_vae)
					data_first = data_train[indices_vae_mb[:batch_size_vae]].to(self.device)
					data_second = data_train[indices_vae_mb[batch_size_vae:]].to(self.device)
					data_use = torch.logical_or(data_first, data_second).to(self.device, dtype=torch.float)
				else:
					indices_vae_mb = random.sample(range(n_train), batch_size_vae)
					data_use = data_train[indices_vae_mb].to(self.device, dtype=torch.float)

				output, mean, logvar, sample = self.genovae_model(data_use)
				vaeLoss = vae_loss(output, data_use, mean, logvar)

				optimizer.zero_grad()
				vaeLoss.backward()
				optimizer.step()
				train_vae_loss += vaeLoss.item()

			train_vae_loss /= n_vae_train

			# validation
			self.genovae_model.eval()
			test_vae_loss = 0

			with torch.no_grad():
				for batch_idx, (_, batch_prop, _) in enumerate(test_prop_loader):
					# vae alone
					check = np.random.sample(1)[0]
					if check > 0.5:
						indices_vae_mb = random.sample(range(n_test), 2 * batch_size_vae)
						data_first = data_test[indices_vae_mb[:batch_size_vae]].to(self.device)
						data_second = data_test[indices_vae_mb[batch_size_vae:]].to(self.device)
						data_use = torch.logical_or(data_first, data_second).to(self.device, dtype=torch.float)
					else:
						indices_vae_mb = random.sample(range(n_test), batch_size_vae)
						data_use = data_test[indices_vae_mb].to(self.device, dtype=torch.float)

					output, mean, logvar, sample = self.genovae_model(data_use)
					vaeLoss = vae_loss(output, data_use, mean, logvar)

					test_vae_loss += vaeLoss.item()

				test_vae_loss /= n_vae_test
				training_time += (time.time() - begin)

			train_vae_loss_list.append(train_vae_loss)
			test_vae_loss_list.append(test_vae_loss)

			if path_save is not None:

				if model_save_per_epochs is not None:

					if epoch % model_save_per_epochs == 0:
						torch.save(self.genovae_model.state_dict(),
								   os.path.join(path_save, "model_params_" + str(epoch) + ".pt"))

		if path_save is not None:

			pd.DataFrame(train_vae_loss_list).to_csv(os.path.join(path_save, "train_vae_loss.csv"))
			pd.DataFrame(test_vae_loss_list).to_csv(os.path.join(path_save, "test_vae_loss.csv"))
			torch.save(self.genovae_model.state_dict(),
					   os.path.join(path_save, "model_params_final_" + str(epoch) + ".pt"))


class GenotypeVAEZLZFineTune:

	"""
	training module for GenotypeVAE with regularization on the Laplacian matrix L
	"""

	def __init__(self, genovae_model, Lmatrix, perturbToOnehot, data_tune_onehot, device):
		super().__init__()
		# Laplacian matrix L
		self.genovae_model = genovae_model
		self.Lmatrix = Lmatrix
		self.perturbToOnehot = perturbToOnehot
		self.data_tune_onehot = data_tune_onehot
		self.device = device

	def extractOneHot(self, perturb_data, batch_prop):
		indx = []

		for i in range(batch_prop.shape[0]):
			indx.append(self.perturbToOnehot[perturb_data[batch_prop[i].item()]])
		return torch.tensor(self.data_tune_onehot[indx])

	def zLz_loss(self, z, Lsub):
		zLz = torch.matmul(torch.matmul(torch.transpose(z, 0, 1), Lsub), z)
		return torch.trace(zLz)

	def extractLmatrix(self, perturb_data, batch_prop):
		indx = []

		for i in range(batch_prop.shape[0]):
			indx.append(self.perturbToOnehot[perturb_data[batch_prop[i].item()]])

		output = np.zeros((len(indx), len(indx)))

		for i in range(len(indx)):
			idex = indx[i]

			for j in range(len(indx)):
				jdex = indx[j]
				output[i, j] = self.Lmatrix[idex, jdex]
				if i != j:
					output[j, i] = self.Lmatrix[idex, jdex]
		return torch.tensor(output)

	def train_np(self, epochs, data_vae_onehot, perturb_with_onehot, lambda_zlz = 1.0, lr_vae = 0.0001,
				 lr_vaezlz = 0.0001, batch_size_vae = 128, batch_size_vaezlz = 128, model_save_per_epochs = None, path_save = None):

		optimizer = optim.Adam(self.genovae_model.parameters(), lr = lr_vae)
		optimizer_tune = optim.Adam(self.genovae_model.parameters(), lr = lr_vaezlz)

		# data loaders
		random_state = np.random.RandomState(seed = 123)
		permutation = random_state.permutation(len(data_vae_onehot))
		n_train = int(len(data_vae_onehot) * 0.8)
		n_test = len(data_vae_onehot) - n_train

		data_train, data_test = torch.tensor(data_vae_onehot[permutation[:n_train]]), torch.tensor(data_vae_onehot[permutation[n_train:]])

		# ZLZ part
		n_vae = perturb_with_onehot.shape[0]
		n_vae_train = int(n_vae * 0.8)
		n_vae_test = n_vae - n_vae_train

		random_vae_state = np.random.RandomState(seed=123)
		permutation_vae = random_vae_state.permutation(n_vae)
		indices_vae_test, indices_vae_train = permutation_vae[:n_vae_test], permutation_vae[n_vae_test:]
		######

		idx_with_onehot_train = torch.tensor(np.array([indices_vae_train]).reshape(-1))
		idx_with_onehot_test = torch.tensor(np.array([indices_vae_test]).reshape(-1))

		train_prop_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(idx_with_onehot_train,
									 idx_with_onehot_train),
			batch_size = batch_size_vaezlz,
			shuffle = True)

		test_prop_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(idx_with_onehot_test,
									 idx_with_onehot_test),
			batch_size = batch_size_vaezlz,
			shuffle = True)

		# train
		train_loss_list, test_loss_list = [], []
		train_vae_loss_list, test_vae_loss_list = [], []

		training_time = 0

		for epoch in range(1, epochs + 1):

			begin = time.time()
			self.genovae_model.train()
			train_loss = 0
			train_vae_loss = 0


			for batch_idx, (_, batch_prop, _) in enumerate(train_prop_loader):
				batch_anno = self.extractOneHot(perturb_with_onehot, batch_prop)
				batch_anno = batch_anno.float().to(self.device)

				# operations0
				output, mean, logvar, sample = self.genovae_model(batch_anno)

				# loss
				vaeLoss = vae_loss(output, batch_anno, mean, logvar)

				# zLz
				Lsub = self.extractLmatrix(perturb_with_onehot, batch_prop)
				mseLoss = lambda_zlz * self.zLz_loss(sample.float().to(self.device), Lsub.float().to(self.device))

				optimizer_tune.zero_grad()
				loss = vaeLoss + mseLoss
				loss.backward()
				optimizer_tune.step()

				train_loss += vaeLoss.item() + mseLoss.item()

				# vae alone
				check = np.random.sample(1)[0]
				if check > 0.5:
					indices_vae_mb = random.sample(range(n_train), 2 * batch_size_vae)
					data_first = data_train[indices_vae_mb[:batch_size_vae]].to(self.device)
					data_second = data_train[indices_vae_mb[batch_size_vae:]].to(self.device)
					data_use = torch.logical_or(data_first, data_second).to(self.device, dtype=torch.float)
				else:
					indices_vae_mb = random.sample(range(n_train), batch_size_vae)
					data_use = data_train[indices_vae_mb].to(self.device, dtype=torch.float)

				output, mean, logvar, sample = self.genovae_model(data_use)
				vaeLoss = vae_loss(output, data_use, mean, logvar)

				optimizer.zero_grad()
				vaeLoss.backward()
				optimizer.step()
				train_vae_loss += vaeLoss.item()


			train_loss /= n_vae_train
			train_vae_loss /= n_vae_train

			# validation
			self.genovae_model.eval()
			test_loss = 0
			test_vae_loss = 0

			with torch.no_grad():
				for batch_idx, (_, batch_prop, _) in enumerate(test_prop_loader):

					batch_anno = self.extractOneHot(perturb_with_onehot, batch_prop)
					batch_anno = batch_anno.float().to(self.device)

					output, mean, logvar, sample = self.genovae_model(batch_anno)

					vaeLoss = vae_loss(output, batch_anno, mean, logvar)

					# zLz
					Lsub = self.extractLmatrix(perturb_with_onehot, batch_prop)
					mseLoss = lambda_zlz * self.zLz_loss(sample.float().to(self.device), Lsub.float().to(self.device))
					test_loss += vaeLoss.item() + mseLoss.item()

					# vae alone
					check = np.random.sample(1)[0]
					if check > 0.5:
						indices_vae_mb = random.sample(range(n_test), 2 * batch_size_vae)
						data_first = data_test[indices_vae_mb[:batch_size_vae]].to(self.device)
						data_second = data_test[indices_vae_mb[batch_size_vae:]].to(self.device)
						data_use = torch.logical_or(data_first, data_second).to(self.device, dtype=torch.float)
					else:
						indices_vae_mb = random.sample(range(n_test), batch_size_vae)
						data_use = data_test[indices_vae_mb].to(self.device, dtype=torch.float)

					output, mean, logvar, sample = self.genovae_model(data_use)
					vaeLoss = vae_loss(output, data_use, mean, logvar)

					test_vae_loss += vaeLoss.item()

				test_loss /= n_vae_test
				test_vae_loss /= n_vae_test

				training_time += (time.time() - begin)

			train_loss_list.append(train_loss)
			test_loss_list.append(test_loss)

			train_vae_loss_list.append(train_vae_loss)
			test_vae_loss_list.append(test_vae_loss)

			if path_save is not None:

				if model_save_per_epochs is not None:

					if epoch % model_save_per_epochs == 0:
						torch.save(self.genovae_model.state_dict(),
								   os.path.join(path_save, "model_params_" + str(epoch) + ".pt"))

		if path_save is not None:
			pd.DataFrame(train_loss_list).to_csv(os.path.join(path_save, "train_loss.csv"))
			pd.DataFrame(test_loss_list).to_csv(os.path.join(path_save, "test_loss.csv"))
			pd.DataFrame(train_vae_loss_list).to_csv(os.path.join(path_save, "train_vae_loss.csv"))
			pd.DataFrame(test_vae_loss_list).to_csv(os.path.join(path_save, "test_vae_loss.csv"))
			torch.save(self.genovae_model.state_dict(),
					   os.path.join(path_save, "model_params_final_" + str(epoch) + ".pt"))







	
