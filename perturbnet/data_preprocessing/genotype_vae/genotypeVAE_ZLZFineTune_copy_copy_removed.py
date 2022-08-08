#!/usr/bin/python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim


import scvi 
import ot
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



def zLz_loss(z, Lmatrix):

	zLz = torch.matmul(torch.matmul(torch.transpose(z, 0, 1), Lmatrix), z)
	return torch.trace(zLz)


def extractLmatrix(Lmatrix, data_perturb_dict, perturb_data, batch_prop):
	indx = []
	for i in range(batch_prop.shape[0]):
		indx.append(data_perturb_dict[perturb_data[batch_prop[i].item()]])
	#return torch.tensor(Lmatrix[indx, indx])
	output = np.zeros((len(indx), len(indx)))

	for i in range(len(indx)):
		idex = indx[i]
		for j in range(i, len(indx)):
			jdex = indx[j]
			output[i, j] = Lmatrix[idex, jdex]
			if i != j:
				output[j, i] = Lmatrix[idex, jdex]
	return torch.tensor(output)


if __name__ == "__main__":
	
	path_save = 'output_latent10_epoch300_GenotypeVAE_1em4_sumLoss_ZLZMeanFineTune_lambda01_50removed' 
	# data
	lambda_weight = 1.0
	path_turbo = '/nfs/turbo/umms-welchjd/hengshi/perturb_gan/data/GI/wass_distance'
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

	data_train, data_test = torch.tensor(data_npz[permutation[:n_train]]), torch.tensor(data_npz[permutation[n_train:]])


	# perturbation information data
	cond_all = pd.read_csv(os.path.join(path_data_GI, 'cell_meta_Annotated_KeptPerturb_RAW_LibraryCheck_PyThon37.csv'))
	null_per_category = ['ctrl/ctrl', 'ctrl1/ctrl', 'ctrl10/ctrl', 'ctrl11/ctrl']
	single_null_per_category = ['ctrl', 'ctrl1', 'ctrl10', 'ctrl11']

	perturb_list = list(cond_all['perturbation'])
	perturb1_list = []
	perturb2_list = []
	for i in range(len(perturb_list)):
		per1, per2 = perturb_list[i].split("/")
		perturb1_list.append(per1)
		perturb2_list.append(per2)

	perturb1_list_collect = ['ctrl' if i in single_null_per_category else i for i in perturb1_list]
	perturb2_list_collect = ['ctrl' if i in single_null_per_category else i for i in perturb2_list]

	perturb_list_collect = [perturb1_list_collect[i] + "/" + perturb2_list_collect[i] for i in range(len(perturb2_list_collect))]
	perturb_data_pd = pd.Series(perturb_list_collect).str.get_dummies("/")
	perturb_data = perturb_data_pd.values.astype('float64')

	perturb_data_pd_rowunique = perturb_data_pd.drop_duplicates()
	perturb_col_names = np.array(list(perturb_data_pd_rowunique.columns))
	perturb_with_onehot = np.array(perturb_list_collect)[indices_with_onehot]

	# remove 50 perturbations 
	removed_20_pers = np.load(os.path.join(path_data_GI, "GI_50RemovedPerturbs_KeptPerturb_RAW_LibraryCheck_PyThon37.npy"))
	removed_20_syn_pers = [i.split("/")[1] + '/' + i.split("/")[0] for i in removed_20_pers]
	removed_all_pers = np.append(removed_20_pers, removed_20_syn_pers)

	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	anno_all_cells = anno_all_cells[kept_indices]
	perturb_with_onehot = perturb_with_onehot[kept_indices]

	# L matrix 
	Lmatrix = np.load(os.path.join(path_turbo, 'lap_1_mat_meanRep.npy'))
	data_perturb = pd.read_csv(os.path.join(path_turbo, 'PerturbInfo.csv'))

	data_perturb_dict = {}
	for i in range(data_perturb.shape[0]):
		data_perturb_dict[data_perturb['Perturb1'][i]] = data_perturb['Unnamed: 0'][i]
		data_perturb_dict[data_perturb['Perturb2'][i]] = data_perturb['Unnamed: 0'][i]
	

	batch_size_small = 128
	n_vae = anno_all_cells.shape[0]
	n_vae_train = int(n_vae * 0.8)
	n_vae_test = n_vae - n_vae_train

	# a random split of train and validation datasets
	random_vae_state = np.random.RandomState(seed = 123)
	permutation_vae = random_vae_state.permutation(n_vae)
	indices_vae_test, indices_vae_train = permutation_vae[:n_vae_test], permutation_vae[n_vae_test:]
	anno_train = torch.tensor(anno_all_cells[indices_vae_train])
	anno_test = torch.tensor(anno_all_cells[indices_vae_test])

	idx_with_onehot_train = torch.tensor(np.array([indices_vae_train]).reshape(-1))
	idx_with_onehot_test =  torch.tensor(np.array([indices_vae_test]).reshape(-1))

	train_prop_loader = torch.utils.data.DataLoader(
		ConcatDatasetWithIndices(anno_train, 
			idx_with_onehot_train),
		batch_size = batch_size_small, 
		shuffle = True)

	test_prop_loader = torch.utils.data.DataLoader(
		ConcatDatasetWithIndices(anno_test, 
			idx_with_onehot_test),
		batch_size = batch_size_small, 
		shuffle = True)

	# model 
	torch.manual_seed(42)

	epochs = 300
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model_c = GenoVAE().to(device)
	optimizer = optim.Adam(model_c.parameters(), lr = 0.0001)
	model_c.load_state_dict(torch.load(path_model, map_location = device))


	def train(epoch):

		# train
		model_c.train()
		train_loss = 0
		train_vae_loss = 0
		
		for batch_idx, (batch_anno, batch_prop, indices) in enumerate(train_prop_loader):
			
			batch_anno = batch_anno.float().to(device)
			output, mean, logvar, sample = model_c(batch_anno)
			vaeLoss = vae_loss(output, batch_anno, mean, logvar)

			# zLz
			Lsub = extractLmatrix(Lmatrix, data_perturb_dict, perturb_with_onehot, batch_prop)
			mseLoss = lambda_weight * zLz_loss(sample.float().to(device), Lsub.float().to(device))
			loss = vaeLoss + mseLoss
			#loss = mseLoss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += vaeLoss.item() + mseLoss.item()
			#train_loss += mseLoss.item()
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

			output, mean, logvar, sample = model_c(data_use)
			vaeLoss = vae_loss(output, data_use, mean, logvar)

			optimizer.zero_grad()
			vaeLoss.backward()
			optimizer.step()
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
				output, mean, logvar, sample = model_c(batch_anno)
				vaeLoss = vae_loss(output, batch_anno, mean, logvar)

				# zLz
				Lsub = extractLmatrix(Lmatrix, data_perturb_dict, perturb_with_onehot, batch_prop)
				mseLoss = lambda_weight * zLz_loss(sample.float().to(device), Lsub.float().to(device))
				loss = vaeLoss + mseLoss
				#loss = mseLoss
				test_loss += vaeLoss.item() + mseLoss.item()
				#test_loss += mseLoss.item()
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

				output, mean, logvar, sample = model_c(data_use)
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

