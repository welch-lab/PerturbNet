#!/usr/bin/python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

import scipy
from scipy.sparse import load_npz
import numpy as np
import pandas as pd


class GenoVAE(nn.Module):
	def __init__(self):

		super(GenoVAE, self).__init__()

		# self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
		# self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
		# self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
		# encoder
		self.linear_1 = nn.Linear(15988, 512)
		self.linear_2 = nn.Linear(512, 256)
		self.linear_3_mu = nn.Linear(256, 32)
		self.linear_3_std = nn.Linear(256, 32)
		
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)

		# decoder
		self.linear_4 = nn.Linear(32, 256)
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



def vae_loss(x_decoded_mean, x, z_mean, z_logvar):

	xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average = True)
	kl_loss = -0.5 * torch.mean(torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), 1))
	return xent_loss + kl_loss


if __name__ == "__main__":
	
	path_save = 'output' 
	# data
	path_data = '/nfs/turbo/umms-welchjd/hengshi/GAN/data/Genotype/sparse_gene_anno_matrix.npz'
	data_npz = load_npz(path_data).toarray()

	random_state = np.random.RandomState(seed = 123)
	permutation = random_state.permutation(len(data_npz))
	n_train = int(len(data_npz) * 0.8)
	n_test = len(data_npz) - n_train
	batch_size = 256

	data_train, data_test = data_npz[permutation[:n_train]], data_npz[permutation[n_train:]]

	data_train = torch.utils.data.TensorDataset(torch.from_numpy(data_train))
	data_test = torch.utils.data.TensorDataset(torch.from_numpy(data_test))
	train_loader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, shuffle = True)
	test_loader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, shuffle = True)

	# model 
	torch.manual_seed(42)

	epochs = 300
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = GenoVAE().to(device)
	optimizer = optim.Adam(model.parameters(), lr = 0.1)


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
				train_loss += loss.item() * data_use.shape[0]
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
				train_loss += loss.item() * data.shape[0]
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

					test_loss += tloss.item() * data_use.shape[0]
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
					test_loss += tloss.item() * data.shape[0]
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

