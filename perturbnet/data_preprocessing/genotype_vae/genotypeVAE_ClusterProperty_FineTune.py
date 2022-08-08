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
from genotypeVAE_sumLoss import * 
import random


class ConcatDatasetWithIndices(torch.utils.data.Dataset):
	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, i):
		return tuple([d[i] for d in self.datasets] + [i])

	def __len__(self):
		return min(len(d) for d in self.datasets)


class GenoVAEFineTune(nn.Module):
	def __init__(self, GenoVAE):
		super().__init__()
		self.GenoVAE = GenoVAE
		self.linear_lat = nn.Linear(10, 10)
		self.bn_lat = nn.BatchNorm1d(10)
		# decoder
		self.linear_lat_out = nn.Linear(10, 6)
		# activations		
		self.relu = nn.ReLU()
		#self.latent_property()

	def forward(self, x):
		z_mean, z_logvar = self.GenoVAE.encode(x)
		self.GenoVAE.z = self.GenoVAE.sampling(z_mean, z_logvar)
		z = self.linear_lat(self.GenoVAE.z)
		z = self.bn_lat(z)
		z = self.relu(z)
		self.property = F.softmax(self.linear_lat_out(z), dim = 1)

		return self.GenoVAE.decode(self.GenoVAE.z), z_mean, z_logvar, self.GenoVAE.z, self.property


def mse_loss(input, target):

	loss = nn.MSELoss(size_average = False)
	output = loss(input, target)
	return output


if __name__ == "__main__":
	
	path_save = 'output_latent10_epoch300_GenotypeVAE_1em4_sumLoss_ClusterPropertyFineTune_lambda0' 
	# data
	path_turbo = '/nfs/turbo/umms-welchjd/hengshi/perturb_gan/data/GI/cluster_property'
	path_data = '/nfs/turbo/umms-welchjd/hengshi/GAN/data/Genotype/sparse_gene_anno_matrix.npz'
	path_model = "/scratch/welchjd_root/welchjd/hengshi/GAN/perturb_gan/genotype_vae/output_latent10_epoch300_GenotypeVAE_1em4_sumLoss/model_params.pt"
	path_data_GI = '/nfs/turbo/umms-welchjd/hengshi/perturb_gan/data/GI/'
	data_npz = load_npz(path_data).toarray()

	# annotation
	anno_all_cells_all = np.load(os.path.join(path_data_GI, "onehot_per_cell_meta_Annotated_KeptPerturb_RAW_LibraryCheck_PyThon37.npy"))
	indices_with_onehot_data = pd.read_csv(os.path.join(path_data_GI, 'indices_with_onehot_cell_meta_Annotated_KeptPerturb_RAW_LibraryCheck_PyThon37.csv'))
	indices_with_onehot = list(indices_with_onehot_data.iloc[:, 1])
	anno_all_cells = anno_all_cells_all[indices_with_onehot]


	random_state = np.random.RandomState(seed = 123)
	permutation = random_state.permutation(len(data_npz))
	n_train = int(len(data_npz) * 0.8)
	n_test = len(data_npz) - n_train
	batch_size = 128

	#data_train, data_test = data_npz[permutation[:n_train]], data_npz[permutation[n_train:]]
	data_train, data_test = torch.tensor(data_npz[permutation[:n_train]]), torch.tensor(data_npz[permutation[n_train:]])
	data_geno = np.load(os.path.join(path_turbo, '6_k_means_clusters_All_Annotated_KeptPerturb_RAW_LibraryCheck_PyThon37.npy'))
	data_geno = data_geno[indices_with_onehot]

	batch_size_small = 128
	n_vae = data_geno.shape[0]
	n_vae_train = int(n_vae * 0.8)
	n_vae_test = n_vae - n_vae_train

	# a random split of train and validation datasets
	random_vae_state = np.random.RandomState(seed = 123)
	permutation_vae = random_vae_state.permutation(n_vae)
	indices_vae_test, indices_vae_train = permutation_vae[:n_vae_test], permutation_vae[n_vae_test:]
	anno_train = torch.tensor(anno_all_cells[indices_vae_train])
	anno_test = torch.tensor(anno_all_cells[indices_vae_test])

	prop_geno_train = torch.tensor(data_geno[indices_vae_train])
	prop_geno_test = torch.tensor(data_geno[indices_vae_test])

	train_prop_loader = torch.utils.data.DataLoader(
		ConcatDatasetWithIndices(anno_train, 
			prop_geno_train),
		batch_size = batch_size_small, 
		shuffle = True)

	test_prop_loader = torch.utils.data.DataLoader(
		ConcatDatasetWithIndices(anno_test, 
			prop_geno_test),
		batch_size = batch_size_small, 
		shuffle = True)

	# model 
	torch.manual_seed(42)

	epochs = 300
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = GenoVAE().to(device)
	optimizer_vae = optim.Adam(model.parameters(), lr = 0.0001)

	model.load_state_dict(torch.load(path_model, map_location = device))
	model_c = GenoVAEFineTune(GenoVAE = model).to(device)
	optimizer = optim.Adam(model_c.parameters(), lr = 0.0001)


	def train(epoch):

		# train
		model_c.train()
		train_loss = 0
		train_vae_loss = 0
		
		for batch_idx, (batch_anno, batch_prop, indices) in enumerate(train_prop_loader):
			
			batch_anno = batch_anno.float().to(device)
			output, mean, logvar, sample, prop = model_c(batch_anno)
			vaeLoss = vae_loss(output, batch_anno, mean, logvar)
			mseLoss = mse_loss(batch_prop.float().to(device), prop)
			#loss = vaeLoss + mseLoss
			loss = vaeLoss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#train_loss += vaeLoss.item() + mseLoss.item()
			train_loss += vaeLoss.item()
			# update the vae alone
			check = np.random.sample(1)[0]
			if check > 0.5:
				indices_vae_mb = random.sample(range(n_train), 2 * batch_size)

				data_first = data_train[indices_vae_mb[:batch_size]].to(device)
				data_second = data_train[indices_vae_mb[batch_size:]].to(device)
				data_use = torch.logical_or(data_first, data_second).to(device, dtype = torch.float)

			else:
				indices_vae_mb = random.sample(range(n_train), batch_size)
				data_use = data_train[indices_vae_mb].to(device, dtype = torch.float)

			output, mean, logvar, sample, _ = model_c(data_use)
			vaeLoss = vae_loss(output, data_use, mean, logvar)

			optimizer_vae.zero_grad()
			vaeLoss.backward()
			optimizer_vae.step()
			train_vae_loss += vaeLoss.item()

		
		train_loss /= n_vae_train 
		train_vae_loss /= n_vae_train 

		# validation
		model_c.eval()
		test_loss = 0
		test_vae_loss = 0

		with torch.no_grad():
			for batch_idx, (batch_anno, batch_prop, indices) in enumerate(test_prop_loader):

				batch_anno = batch_anno.float().to(device)
				output, mean, logvar, sample, prop = model_c(batch_anno)
				vaeLoss = vae_loss(output, batch_anno, mean, logvar)
				mseLoss = mse_loss(batch_prop.float().to(device), prop)
				#loss = vaeLoss + mseLoss
				loss = vaeLoss
				#test_loss += vaeLoss.item() + mseLoss.item()
				test_loss += vaeLoss.item()
				# update the vae alone
				check = np.random.sample(1)[0]
				if check > 0.5:
					indices_vae_mb = random.sample(range(n_test), 2 * batch_size)

					data_first = data_test[indices_vae_mb[:batch_size]].to(device)
					data_second = data_test[indices_vae_mb[batch_size:]].to(device)
					data_use = torch.logical_or(data_first, data_second).to(device, dtype = torch.float)

				else:
					indices_vae_mb = random.sample(range(n_test), batch_size)
					data_use = data_test[indices_vae_mb].to(device, dtype = torch.float)

				output, mean, logvar, sample, _ = model_c(data_use)
				vaeLoss = vae_loss(output, data_use, mean, logvar)

				test_vae_loss += vaeLoss.item()

			
			test_loss /= n_vae_test 
			test_vae_loss /= n_vae_test 


		return train_loss, test_loss, train_vae_loss, test_vae_loss

	train_loss_list, test_loss_list = [], []
	train_vae_loss_list, test_vae_loss_list = [], []
	for epoch in range(1, epochs + 1):
		train_loss, test_loss, train_vae_loss, test_vae_loss = train(epoch)
		train_loss_list.append(train_loss)
		test_loss_list.append(test_loss)

		train_vae_loss_list.append(train_vae_loss)
		test_vae_loss_list.append(test_vae_loss)



	pd.DataFrame(train_loss_list).to_csv(os.path.join(path_save, "train_loss.csv"))
	pd.DataFrame(test_loss_list).to_csv(os.path.join(path_save, "test_loss.csv"))
	pd.DataFrame(train_vae_loss_list).to_csv(os.path.join(path_save, "train_vae_loss.csv"))
	pd.DataFrame(test_vae_loss_list).to_csv(os.path.join(path_save, "test_vae_loss.csv"))

	torch.save(model_c.state_dict(), os.path.join(path_save, "model_params.pt"))

