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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class GenoVAE(nn.Module):
	def __init__(self):

		super(GenoVAE, self).__init__()

		# self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
		# self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
		# self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
		self.linear_0 = nn.Linear(15988, 512)
		self.linear_012 = nn.Linear(512, 256)
		self.linear_1 = nn.Linear(256, 32)
		self.linear_2 = nn.Linear(256, 32)

		self.linear_3 = nn.Linear(32, 256)
		self.linear_34 = nn.Linear(256, 512)
		# self.gru = nn.GRU(292, 501, 3, batch_first=True)
		self.linear_4 = nn.Linear(512, 15988)
		
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()

	def encode(self, x):
		# x = self.relu(self.conv_1(x))
		# x = self.relu(self.conv_2(x))
		# x = self.relu(self.conv_3(x))
		# x = x.view(x.size(0), -1)
		x = self.relu(self.linear_0(x))
		x = F.selu(self.linear_012(x))
		return self.linear_1(x), self.linear_2(x)

	def sampling(self, z_mean, z_logvar):
		epsilon = 1e-2 * torch.randn_like(z_logvar)
		return torch.exp(0.5 * z_logvar) * epsilon + z_mean

	def decode(self, z):
		z = F.selu(self.linear_3(z))
		out_reshape = F.selu(self.linear_34(z))
		# z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
		# output, hn = self.gru(z)
		# out_reshape = output.contiguous().view(-1, output.size(-1))
		y0 = F.softmax(self.linear_4(out_reshape), dim = 1)
		y = y0.contiguous()
		return y

	def forward(self, x):
		z_mean, z_logvar = self.encode(x)
		self.z = self.sampling(z_mean, z_logvar)
		return self.decode(self.z), z_mean, z_logvar, self.z


def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
	xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
	kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
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
	batch_size = 128

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
	optimizer = optim.Adam(model.parameters())


	def train(epoch):
		model.train()
		train_loss = 0
		for batch_idx, data in enumerate(train_loader):
			data = data[0].to(device, dtype = torch.float)

			check = np.random.sample(1)[0]
			data_len = data.shape[0]
			
			if check > 0.5 and data_len % 2 == 0:
				half_len = int(data_len / 2)
				data_first, data_second = data[:half_len], data[half_len]
				data = torch.logical_or(data_first, data_second).to(device, dtype = torch.float)

			optimizer.zero_grad()
			output, mean, logvar, _ = model(data)
			
			if batch_idx == 0:
				inp = data.cpu().numpy()
				outp = output.cpu().detach().numpy()
				lab = data.cpu().numpy()
				#print("Input:")
				#print(decode_smiles_from_indexes(map(from_one_hot_array, inp[0]), charset))
				#print("Label:")
				#print(decode_smiles_from_indexes(map(from_one_hot_array, lab[0]), charset))
				# sampled = outp[0].reshape(1, 120, len(charset)).argmax(axis=2)[0]
				# print("Output:")
				# print(decode_smiles_from_indexes(sampled, charset))
				model.eval()
				test_loss = 0
				with torch.no_grad():
					for batch_test_idx, testdata in enumerate(test_loader):
						testdata = testdata[0].to(device, dtype = torch.float)

						check = np.random.sample(1)[0]
						data_len = testdata.shape[0]

						if check > 0.5 and data_len % 2 == 0:
							half_len = int(data_len / 2)
							data_first, data_second = testdata[:half_len], testdata[half_len]
							testdata = torch.logical_or(data_first, data_second).to(device, dtype = torch.float)
						
						output_test, mean_test, logvar_test, _ = model(testdata)
						tloss = vae_loss(output_test, testdata, mean_test, logvar_test)
						test_loss += tloss.item() * testdata.shape[0]
					test_loss /= len(test_loader.dataset)
				model.train()

			loss = vae_loss(output, data, mean, logvar)
			loss.backward()
			train_loss += loss.item() * data.shape[0]
			optimizer.step()
		train_loss /= len(train_loader.dataset)
	#         if batch_idx % 100 == 0:
	#             print(f'{epoch} / {batch_idx}\t{loss:.4f}')
		print('train', train_loss / len(train_loader.dataset))
		return train_loss, test_loss

	train_loss_list, test_loss_list = [], []
	for epoch in range(1, epochs + 1):
		train_loss, test_loss = train(epoch)
		train_loss_list.append(train_loss)
		test_loss_list.append(test_loss)
	pd.DataFrame(train_loss_list).to_csv(os.path.join(path_save, "train_loss.csv"))
	pd.DataFrame(test_loss_list).to_csv(os.path.join(path_save, "test_loss.csv"))
	torch.save(model.state_dict(), os.path.join(path_save, "model_params.pt"))

