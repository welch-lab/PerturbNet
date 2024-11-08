#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import time

def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
	"""
	ELBO loss of VAE
	"""

	xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average = False)
	kl_loss = -0.5 * torch.sum(torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), 1))
	return xent_loss + kl_loss


class ConcatDatasetWithIndices(torch.utils.data.Dataset):
	"""
	data structure with sample indices of two datasets
	"""
	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, i):
		return tuple([d[i] for d in self.datasets] + [i])

	def __len__(self):
		return min(len(d) for d in self.datasets)

class ChemicalVAE(nn.Module):
	"""
	ChemicalVAE module
	"""
	def __init__(self, n_char, 
				 z_dim = 196, 
				 n_conv = int(8 * 1.15875438383), 
				 k_conv = int(8 * 1.1758149644), 
				 n_conv2 = int(8 * 1.15875438383 ** 2), 
				 k_conv2 = int(8 * 1.1758149644 ** 2), 
				 prob_drop = 0.082832929704794792, 
				 max_len = 120):

		super(ChemicalVAE, self).__init__()

		self.conv_1 = nn.Conv1d(in_channels = max_len, 
								out_channels = n_conv, 
								kernel_size = k_conv)

		self.conv_2 = nn.Conv1d(in_channels = n_conv, 
								out_channels = n_conv, 
								kernel_size = k_conv) 

		self.conv_3 = nn.Conv1d(in_channels = n_conv, 
								out_channels = n_conv2, 
								kernel_size = k_conv2) 


		self.bnConv1 = nn.BatchNorm1d(n_conv)
		self.bnConv2 = nn.BatchNorm1d(n_conv)
		self.bnConv3 = nn.BatchNorm1d(n_conv2)
		
		
		self.linear_0 = nn.Linear(90, z_dim)
		self.linear_1 = nn.Linear(z_dim, z_dim)
		self.linear_2 = nn.Linear(z_dim, z_dim)

		self.dropout1 = nn.Dropout(p = prob_drop)
		self.bn1 = nn.BatchNorm1d(z_dim)

		self.dropout2 = nn.Dropout(p = prob_drop)
		self.bn2 = nn.BatchNorm1d(z_dim)


		self.linear_3 = nn.Linear(z_dim, z_dim)
		self.gru = nn.GRU(input_size = z_dim, 
						  hidden_size = 488, 
						  num_layers = 3, 
						  batch_first = True)
		self.linear_4 = nn.Linear(488, n_char)
		
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
		self.tanh = nn.Tanh()
		self.max_len = max_len

	def encode(self, x):
		"""
		encode one-hot encodings of chemicals
		"""
		x = self.tanh(self.conv_1(x))
		x = self.bnConv1(x)

		x = self.tanh(self.conv_2(x))
		x = self.bnConv2(x)

		x = self.tanh(self.conv_3(x))
		x = self.bnConv3(x)

		x = x.view(x.size(0), -1)
		x = self.tanh(self.linear_0(x))
		x = self.dropout1(x)
		x = self.bn1(x)

		return self.linear_1(x), self.linear_2(x)

	def sampling(self, z_mean, z_logvar):
		"""
		sampled representations from the latent means and stds
		"""
		epsilon =  1e-2 * torch.randn_like(z_logvar)
		return torch.exp(0.5 * z_logvar) * epsilon + z_mean

	def decode(self, z):
		"""
		decode latent representations to one-hot encodings
		"""
		z = self.tanh(self.linear_3(z))
		z = self.dropout2(z)
		z = self.bn2(z)

		z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_len, 1)
		output, hn = self.gru(z)
		output = self.tanh(output)

		out_reshape = output.contiguous().view(-1, output.size(-1))
		y0 = F.softmax(self.linear_4(out_reshape), dim = 1)
		y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
		return y

	def forward(self, x):
		z_mean, z_logvar = self.encode(x)
		self.z = self.sampling(z_mean, z_logvar)
		return self.decode(self.z), z_mean, z_logvar, self.z


class ChemicalVAETrain:
	"""
	training module to train ChemicalVAE
	"""

	def __init__(self, chemvae_model, device):

		super().__init__()
		# Laplacian matrix L
		self.chemvae_model = chemvae_model
		self.device = device

	def train_np(self, 
				 epochs, 
				 data_vae_onehot, 
				 perturb_with_onehot,
				 lr_vae = 0.0001, 
				 batch_size_vae = 128, 
				 batch_size_vaezlz =128, 
				 model_save_per_epochs = None,
				 path_save = None):

		optimizer = optim.Adam(self.chemvae_model.parameters(), lr = lr_vae)

		# all data
		random_state = np.random.RandomState(seed=123)
		permutation = random_state.permutation(len(data_vae_onehot))
		n_train = int(len(data_vae_onehot) * 0.8)
		n_test = len(data_vae_onehot) - n_train

		data_chem_train, data_chem_test = torch.tensor(data_vae_onehot[permutation[:n_train]]), torch.tensor(
			data_vae_onehot[permutation[n_train:]])

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
			batch_size=batch_size_vaezlz,
			shuffle=True)

		# train
		train_vae_loss_list, test_vae_loss_list = [], []
		training_time = 0
		for epoch in range(1, epochs + 1):

			begin = time.time()
			self.chemvae_model.train()
			train_vae_loss = 0

			for batch_idx, (_, batch_prop, _) in enumerate(train_prop_loader):

				# vae alone
				indices_vae_mb = random.sample(range(n_train), batch_size_vae)
				data_use = data_chem_train[indices_vae_mb].to(self.device, dtype=torch.float)

				output, mean, logvar, sample = self.chemvae_model(data_use)
				vaeLoss = vae_loss(output, data_use, mean, logvar)

				optimizer.zero_grad()
				vaeLoss.backward()
				optimizer.step()
				train_vae_loss += vaeLoss.item()

			train_vae_loss /= n_vae_train

			# validation
			self.chemvae_model.eval()
			test_vae_loss = 0

			with torch.no_grad():
				for batch_idx, (_, batch_prop, _) in enumerate(test_prop_loader):

					# vae alone
					indices_vae_mb = random.sample(range(n_test), batch_size_vae)
					data_use = data_chem_test[indices_vae_mb].to(self.device, dtype=torch.float)

					output, mean, logvar, sample = self.chemvae_model(data_use)
					vaeLoss = vae_loss(output, data_use, mean, logvar)

					test_vae_loss += vaeLoss.item()

				test_vae_loss /= n_vae_test
				training_time += (time.time() - begin)

			train_vae_loss_list.append(train_vae_loss)
			test_vae_loss_list.append(test_vae_loss)

			if path_save is not None:

				if model_save_per_epochs is not None:

					if epoch % model_save_per_epochs == 0:
						torch.save(self.chemvae_model.state_dict(),
								   os.path.join(path_save, "model_params_" + str(epoch) + ".pt"))

		if path_save is not None:
			pd.DataFrame(train_vae_loss_list).to_csv(os.path.join(path_save, "train_vae_loss.csv"))
			pd.DataFrame(test_vae_loss_list).to_csv(os.path.join(path_save, "test_vae_loss.csv"))
			torch.save(self.chemvae_model.state_dict(),
					   os.path.join(path_save, "model_params_final_" + str(epoch) + ".pt"))


class ChemicalVAEFineTuneZLZ:

	"""
	training module for ChemicalVAE with regularization on the Laplacian matrix L
	"""
	def __init__(self, chemvae_model, Lmatrix, perturbToOnehot, data_tune_onehot, device):

		super().__init__()
		# Laplacian matrix L
		self.chemvae_model = chemvae_model
		self.Lmatrix = Lmatrix
		self.perturbToOnehot = perturbToOnehot
		self.data_tune_onehot = data_tune_onehot
		self.device = device

	def zLz_loss(self, z, Lsub):
		
		zLz = torch.matmul(torch.matmul(torch.transpose(z, 0, 1), Lsub), z)

		return torch.trace(zLz)


	def extractOneHot(self, perturb_data, batch_prop):
		indx = []
		for i in range(batch_prop.shape[0]):
			indx.append(self.perturbToOnehot[perturb_data[batch_prop[i].item()]])

		return torch.tensor(self.data_tune_onehot[indx])


	def extractLmatrix(self, perturb_data, batch_prop):

		indx = []
		for i in range(batch_prop.shape[0]):
			indx.append(self.perturbToOnehot[perturb_data[batch_prop[i].item()]])

		output = np.zeros((len(indx), len(indx)))

		for i in range(len(indx)):
			idex = indx[i]

			for j in range(i, len(indx)):
				jdex = indx[j]
				output[i, j] = self.Lmatrix[idex, jdex]
				if i != j:
					output[j, i] = self.Lmatrix[idex, jdex]

		return torch.tensor(output)

	def train_np(self, epochs, data_vae_onehot, perturb_with_onehot, lambda_zlz = 1.0, 
				 lr_vae = 0.0001,  lr_vaezlz = 0.0001, batch_size_vae = 128, batch_size_vaezlz = 128, model_save_per_epochs = None, path_save = None):

		optimizer = optim.Adam(self.chemvae_model.parameters(), lr = lr_vae)
		optimizer_tune = optim.Adam(self.chemvae_model.parameters(), lr = lr_vaezlz)
		
		# all data
		random_state = np.random.RandomState(seed = 123)
		permutation = random_state.permutation(len(data_vae_onehot))
		n_train = int(len(data_vae_onehot) * 0.8)
		n_test = len(data_vae_onehot) - n_train

		data_chem_train, data_chem_test = torch.tensor(data_vae_onehot[permutation[:n_train]]), torch.tensor(data_vae_onehot[permutation[n_train:]])

		# ZLZ part

		n_vae = perturb_with_onehot.shape[0]
		n_vae_train = int(n_vae * 0.8)
		n_vae_test = n_vae - n_vae_train

		random_vae_state = np.random.RandomState(seed = 123)
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
			self.chemvae_model.train()
			train_loss = 0
			train_vae_loss = 0
			

			for batch_idx, (_, batch_prop, _) in enumerate(train_prop_loader):
				
				batch_anno = self.extractOneHot(perturb_with_onehot, batch_prop)
				batch_anno = batch_anno.float().to(self.device)

				# operations0
				output, mean, logvar, sample = self.chemvae_model(batch_anno)

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
				indices_vae_mb = random.sample(range(n_train), batch_size_vae)
				data_use = data_chem_train[indices_vae_mb].to(self.device, dtype = torch.float)

				output, mean, logvar, sample = self.chemvae_model(data_use)
				vaeLoss = vae_loss(output, data_use, mean, logvar)

				optimizer.zero_grad()
				vaeLoss.backward()
				optimizer.step()
				train_vae_loss += vaeLoss.item()


			train_loss /= n_vae_train 
			train_vae_loss /= n_vae_train 

			# validation
			self.chemvae_model.eval()
			test_loss = 0
			test_vae_loss = 0

			with torch.no_grad():
				for batch_idx, (_, batch_prop, _) in enumerate(test_prop_loader):

					
					batch_anno = self.extractOneHot(perturb_with_onehot, batch_prop)
					batch_anno = batch_anno.float().to(self.device)

					output, mean, logvar, sample = self.chemvae_model(batch_anno)

					
					vaeLoss = vae_loss(output, batch_anno, mean, logvar)

					# zLz
					Lsub = self.extractLmatrix(perturb_with_onehot, batch_prop)
					mseLoss = lambda_zlz * self.zLz_loss(sample.float().to(self.device), Lsub.float().to(self.device))
					test_loss += vaeLoss.item() + mseLoss.item()

					# vae alone 
					indices_vae_mb = random.sample(range(n_test), batch_size_vae)
					data_use = data_chem_test[indices_vae_mb].to(self.device, dtype = torch.float)

					output, mean, logvar, sample = self.chemvae_model(data_use)
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
						
						torch.save(self.chemvae_model.state_dict(), os.path.join(path_save, "model_params_" + str(epoch) + ".pt"))

		if path_save is not None:
			pd.DataFrame(train_loss_list).to_csv(os.path.join(path_save, "train_loss.csv"))
			pd.DataFrame(test_loss_list).to_csv(os.path.join(path_save, "test_loss.csv"))
			pd.DataFrame(train_vae_loss_list).to_csv(os.path.join(path_save, "train_vae_loss.csv"))
			pd.DataFrame(test_vae_loss_list).to_csv(os.path.join(path_save, "test_vae_loss.csv"))
			torch.save(self.chemvae_model.state_dict(), os.path.join(path_save, "model_params_final_" + str(epoch) + ".pt"))


