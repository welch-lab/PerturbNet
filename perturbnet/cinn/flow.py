#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn

from perturbnet.net2net.modules.flow.loss import NLL
from perturbnet.net2net.ckpt_util import get_ckpt_path
from perturbnet.net2net.modules.util import log_txt_as_img
from perturbnet.net2net.modules.flow.flatflow import *

from perturbnet.pytorch_scvi.distributions import *
from perturbnet.pytorch_scvi.scvi_generate_z import *

import sys
sys.path.append("./../../")
from perturbnet.util import *



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
    

class Net2NetFlow_scVIChemStdFlow(nn.Module):
	"""
	cINN module connecting standardized ChemicalVAE latent space to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data, 
				 cond_stage_data,
				 model_con, 
				 scvi_model, 
				 std_model, 
				 model_to_use = None, 
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.first_stage_data = first_stage_data
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.model_con = model_con
		self.scvi_model = scvi_model
		self.std_model = std_model

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use 
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime


	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	##########################

	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train"):
		"""
		compute the loss value in a batch 
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict



	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr = lr,
							   betas = (0.5, 0.9),
							   amsgrad = True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr = lr,
							   #betas=(0.5, 0.9),
							   amsgrad = False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self, n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()


		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train 

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]


		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_train, 
				cond_stage_train),
			batch_size = batch_size, 
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_test, 
				cond_stage_test),
			batch_size = batch_size, 
			shuffle = False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):
				
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)
				batch_cond_x = batch_cond_x.float().to(device)

				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x)).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]


			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)
				#batch_cond_x = batch_cond_x.float().to(device)
				#batch_cond_x = self.std_model.standardize_z_torch(batch_cond_x.float()).to(device)
				#batch_cond = self.encode_con(batch_cond_x).to(device)

				batch_cond_x = batch_cond_x.float().to(device)
				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x)).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

	def train_trainTestDiffPer(self, perturb_arr, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		begin = time.time()

		perturb_unique = np.unique(perturb_arr)
		n = len(perturb_unique)
		n_train = int(n * 0.8)
		n_test = n - n_train

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_per_test, indices_per_train = permutation[:n_test], permutation[n_test:]
		perturb_train, perturb_test = perturb_unique[indices_per_train], perturb_unique[indices_per_test]

		indices_train = np.where(np.in1d(perturb_arr, perturb_train))[0]
		indices_test = np.where(np.in1d(perturb_arr, perturb_test))[0]

		cond_stage_train = self.cond_stage_data[indices_train]
		cond_stage_test = self.cond_stage_data[indices_test]

		num_train, num_test = len(indices_train), len(indices_test)


		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
									 cond_stage_train),
			batch_size=batch_size,
			shuffle=False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
									 cond_stage_test),
			batch_size=batch_size,
			shuffle=False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_train[indices],
																					 give_mean=False)).float().to(device)
				batch_cond_x = batch_cond_x.float().to(device)
				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x)).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_test[indices],
																					 give_mean=False)).float().to(
					device)
				batch_cond_x = batch_cond_x.float().to(device)
				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x)).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= num_train
			test_loss /= num_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)


	def train_vaeflow(self, n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1):
		"""
		train the net2net model, with train and test validation
		"""

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()


		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train 

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]


		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_train, 
				cond_stage_train),
			batch_size = batch_size, 
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_test, 
				cond_stage_test),
			batch_size = batch_size, 
			shuffle = False)

		optimizer = self.configure_vaeflow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			self.model_con.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):
				
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)
				batch_cond_x = batch_cond_x.float().to(device)

				output, mean, logvar, sample = self.model_con(batch_cond_x)
				batch_cond = sample.to(device)

				vaeLoss = vae_loss(output, batch_cond_x, mean, logvar)
				flowLoss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				
				loss = vaeLoss + flowLoss

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += flowLoss.item() * batch_first.shape[0] + vaeLoss.item()


			self.flow.eval()
			self.model_con.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)
				batch_cond_x = batch_cond_x.float().to(device)

				output, mean, logvar, sample = self.model_con(batch_cond_x)
				batch_cond = sample.to(device)

				vaeLoss = vae_loss(output, batch_cond_x, mean, logvar)
				flowLoss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				
				loss = vaeLoss + flowLoss

				test_loss += flowLoss.item() * batch_first.shape[0] + vaeLoss.item()

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

			

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite: 
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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


class Net2NetFlow_scVIChemStdStatesFlow(nn.Module):
	"""
	cINN module connecting standardized ChemicalVAE latent space and cell state
	covariates to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data,
				 cond_stage_data,
				 model_con,
				 scvi_model,
				 std_model,
				 cell_type,
				 dose,
				 model_to_use=None,
				 ignore_keys=[],
				 first_stage_key="cell",
				 cond_stage_key="cell",
				 interpolate_cond_size=-1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.first_stage_data = first_stage_data
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.model_con = model_con
		self.scvi_model = scvi_model
		self.std_model = std_model
		self.cell_type_onehot = cell_type
		self.dose_onehot = dose

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime

	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	##########################

	def shared_step(self, batch_first, batch_cond, batch_idx, split="train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first  # self.get_input(self.first_stage_key, batch)
		c = batch_cond  # self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split=split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr=lr,
							   # betas=(0.5, 0.9),
							   amsgrad=False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()

		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		dose_cell_onehot = np.concatenate((self.dose_onehot, self.cell_type_onehot), axis = 1)

		first_stage_train, cond_stage_train = dose_cell_onehot[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = dose_cell_onehot[indices_test], self.cond_stage_data[indices_test]

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_train,
									 cond_stage_train),
			batch_size=batch_size,
			shuffle=False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_test,
									 cond_stage_test),
			batch_size=batch_size,
			shuffle=False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_doseCell_x, batch_cond_x, indices) in enumerate(train_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_train[indices],
																					 give_mean=False)).float().to(
					device)
				batch_cond_x = batch_cond_x.float().to(device)

				batch_cond_p = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x)).to(device)
				batch_cond = torch.cat((batch_cond_p, batch_doseCell_x.to(device)), 1)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (batch_doseCell_x, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_test[indices],
																					 give_mean=False)).float().to(
					device)
				# batch_cond_x = batch_cond_x.float().to(device)
				# batch_cond_x = self.std_model.standardize_z_torch(batch_cond_x.float()).to(device)
				# batch_cond = self.encode_con(batch_cond_x).to(device)

				batch_cond_x = batch_cond_x.float().to(device)
				batch_cond_p = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x)).to(device)
				batch_cond = torch.cat((batch_cond_p, batch_doseCell_x.to(device)), 1)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok=overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)

	def load(self, dir_path: str, use_cuda: bool = False, save_use_cuda: bool = False):
		use_cuda = use_cuda and torch.cuda.is_available()
		device = torch.device('cpu') if use_cuda is False else torch.device('cuda')

		model_save_path = os.path.join(dir_path, "model_params.pt")

		if use_cuda and save_use_cuda:
			self.load_state_dict(torch.load(model_save_path))
			self.to(device)
		elif use_cuda and save_use_cuda is False:
			self.load_state_dict(torch.load(model_save_path, map_location="cuda:0"))
			self.to(device)
		else:
			self.load_state_dict(torch.load(model_save_path, map_location=device))


class Net2NetFlow_TFVAEFlow(nn.Module):
	"""
	cINN module connecting standardized ChemicalVAE latent space to VAE latent space
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data, 
				 cond_stage_data, 
				 perturbToOnehotLib,
				 oneHotData, 
				 model_con, 
				 std_model, 
				 sess, enc_ph, z_gen_data_v, is_training, 
				 model_to_use = None, 
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.first_stage_data = first_stage_data
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.perturbToOnehotLib = perturbToOnehotLib 
		self.oneHotData = oneHotData
		self.model_con = model_con
		self.std_model = std_model 
		self.sess = sess
		self.enc_ph = enc_ph
		self.z_gen_data_v = z_gen_data_v
		self.is_training = is_training

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use 
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime

	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec
	##########################

	def get_input(self, key, batch, is_conditioning = False):
		x = batch
		if key in ["caption", "text"]:
			x = list(x[0])
		elif key in ["class"]:
			pass
		else:
			if len(x.shape) == 2:
				x = x[..., None, None]
			elif len(x.shape) == 3:
				x = x[..., None]
			x = x.permute(0, 3, 1, 2).to(memory_format = torch.contiguous_format)
			if is_conditioning:
				if self.interpolate_cond_size > -1:
					x = F.interpolate(x, size=(self.interpolate_cond_size, self.interpolate_cond_size))
		return x

	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train"):
		"""
		compute the loss value in a batch 
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict

	def configure_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	@torch.no_grad()
	def encode(self, x_torch):
		x_np = x_torch.detach().cpu().numpy()
		feed_dict = {self.enc_ph: x_np, self.is_training: False}
		z = self.sess.run(self.z_gen_data_v, feed_dict = feed_dict)
		z_torch = torch.tensor(z).float()
		return z_torch.unsqueeze(-1).unsqueeze(-1)

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	@torch.no_grad()
	def extractOneHot(self, batch_perturb):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[self.cond_stage_data[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])

	@torch.no_grad()
	def extractOneHotUnseen(self, batch_perturb, cond_stage_data_unseen):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[cond_stage_data_unseen[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])

	def train_cinn(self, n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1, auto_save = 0, auto_save_path = "./"):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]


		begin = time.time()


		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train 

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]

		# cond_stage_train = torch.tensor(np.array([cond_stage_train]).reshape(-1))
		# cond_stage_test = torch.tensor(np.array([cond_stage_test]).reshape(-1))

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_train, 
				cond_stage_train),
			batch_size = batch_size, 
			shuffle = True)


		test_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_test, 
				cond_stage_test),
			batch_size = batch_size, 
			shuffle =  True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(train_loader):
				#batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)
				
				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]


			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(test_loader):
				#batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)

				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
            
			if auto_save and epoch % auto_save == 0 and epoch < n_epochs:         
				auto_save_path_model = os.path.join(auto_save_path,"model_params" + str(epoch) + ".pt")
				if not os.path.exists(auto_save_path): 
					os.makedirs(auto_save_path, exist_ok = False)
				torch.save(self.state_dict(), auto_save_path_model)
                
			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)



		self.training_time += (time.time() - begin)


	def train_evaluateUnseenPer(self, data_unseen, cond_stage_data_unseen,  path_save = None,
								n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]


		begin = time.time()


		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train
		n_unseen = len(cond_stage_data_unseen)

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]

		# cond_stage_train = torch.tensor(np.array([cond_stage_train]).reshape(-1))
		# cond_stage_test = torch.tensor(np.array([cond_stage_test]).reshape(-1))

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))
		cond_stage_unseen = torch.tensor(np.array(list(range(len(cond_stage_data_unseen)))).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_train,
						  cond_stage_train),
			batch_size = batch_size,
			shuffle = True)


		test_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_test,
						  cond_stage_test),
			batch_size = batch_size,
			shuffle =  True)

		valid_loader = torch.utils.data.DataLoader(
			ConcatDataset(data_unseen,
						  cond_stage_unseen),
			batch_size = batch_size,
			shuffle = True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		unseen_loss_list = []

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0
			unseen_loss = 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(train_loader):
				#batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)

				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]


			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(test_loader):
				#batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)

				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(valid_loader):
				#batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHotUnseen(batch_cond_x, cond_stage_data_unseen).float().to(device)

				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				unseen_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
			unseen_loss /= n_unseen

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)
			unseen_loss_list.append(unseen_loss)


		if path_save is not None:
			np.save(os.path.join(path_save, "unseen_loss.npy"), np.array(unseen_loss_list))

		self.training_time += (time.time() - begin)

	def train_trainTestDiffPer(self, perturb_arr, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()

		perturb_unique = np.unique(perturb_arr)
		n = len(perturb_unique)
		n_train = int(n * 0.8)
		n_test = n - n_train

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_per_test, indices_per_train = permutation[:n_test], permutation[n_test:]

		perturb_train, perturb_test = perturb_unique[indices_per_train], perturb_unique[indices_per_test]

		indices_train = np.where(np.in1d(perturb_arr, perturb_train))[0]
		indices_test = np.where(np.in1d(perturb_arr, perturb_test))[0]

		first_stage_train = self.first_stage_data[indices_train]
		first_stage_test = self.first_stage_data[indices_test]

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))

		num_train, num_test = len(indices_train), len(indices_test)

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_train,
						  cond_stage_train),
			batch_size = batch_size,
			shuffle = True)

		test_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_test,
						  cond_stage_test),
			batch_size=batch_size,
			shuffle=True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(train_loader):

				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)
				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(test_loader):

				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)
				batch_cond = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= num_train
			test_loss /= num_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite: 
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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


class Net2NetFlow_scVIGenoFlow(nn.Module):
	"""
	cINN module connecting GenotypeVAE latent space to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 cond_stage_data,
				 model_con, 
				 scvi_model,
				 perturbToOnehotLib,
				 oneHotData,     
				 model_to_use = None, 
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.model_con = model_con
		self.scvi_model = scvi_model
		self.perturbToOnehotLib = perturbToOnehotLib
		self.oneHotData = oneHotData

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use 
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime


	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	@torch.no_grad()
	def extractOneHot_scvi(self, batch_perturb):
		indx = []
		for i in range(len(batch_perturb)):
			indx.append(self.perturbToOnehotLib[batch_perturb[i]])
		return torch.tensor(self.oneHotData[indx])

	##########################
	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train"):
		"""
		compute the loss value in a batch 
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr = lr,
							   betas = (0.5, 0.9),
							   amsgrad = True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr = lr,
							   #betas=(0.5, 0.9),
							   amsgrad = False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self, n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		begin = time.time()


		n = self.cond_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train 

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		cond_stage_train = self.cond_stage_data[indices_train]
		cond_stage_test = self.cond_stage_data[indices_test]


		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train, 
				cond_stage_train),
			batch_size = batch_size, 
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
				cond_stage_test),
			batch_size = batch_size, 
			shuffle = False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):
				
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)
				#print(batch_cond_x)          
				batch_cond_x = self.extractOneHot_scvi(batch_cond_x).to(device)
                
				batch_cond = self.encode_con(batch_cond_x.float().to(device))

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]


			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)
				batch_cond_x = self.extractOneHot_scvi(batch_cond_x).to(device)
				batch_cond = self.encode_con(batch_cond_x.float().to(device))

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)


	def train_evaluateUnseenPer(self, anndata_unseen, cond_stage_data_unseen, path_save = None,
								n_epochs = 400, batch_size = 128, lr = 4.5e-6,
								train_ratio = 0.8, seed = 42, start_epoch = 1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		begin = time.time()
		n = self.cond_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train
		n_unseen = len(cond_stage_data_unseen)

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		cond_stage_train = self.cond_stage_data[indices_train]
		cond_stage_test = self.cond_stage_data[indices_test]

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
				cond_stage_train),
			batch_size = batch_size,
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
									 cond_stage_test),
			batch_size = batch_size,
			shuffle = False)

		valid_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_data_unseen,
									 cond_stage_data_unseen),
			batch_size = batch_size,
			shuffle = False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		unseen_loss_list = []

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0
			unseen_loss = 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):
				print(batch_cond_x)
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)

				batch_cond = self.encode_con(batch_cond_x.float().to(device))

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)
				batch_cond = self.encode_con(batch_cond_x.float().to(device))

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			for batch_idx, (_, batch_cond_x, indices) in enumerate(valid_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(adata = anndata_unseen[indices.cpu().detach().numpy(), :].copy(),
																					 give_mean = False)).float().to(device)
				batch_cond = self.encode_con(batch_cond_x.float().to(device))
				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				unseen_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
			unseen_loss /= n_unseen

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)
			unseen_loss_list.append(unseen_loss)

		if path_save is not None:
			np.save(os.path.join(path_save, "unseen_loss.npy"), np.array(unseen_loss_list))

		self.training_time += (time.time() - begin)

	def train_trainTestDiffPer(self, perturb_arr, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		begin = time.time()

		perturb_unique = np.unique(perturb_arr)
		n = len(perturb_unique)
		n_train = int(n * 0.8)
		n_test = n - n_train


		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_per_test, indices_per_train = permutation[:n_test], permutation[n_test:]
		perturb_train, perturb_test = perturb_unique[indices_per_train], perturb_unique[indices_per_test]

		indices_train = np.where(np.in1d(perturb_arr, perturb_train))[0]
		indices_test = np.where(np.in1d(perturb_arr, perturb_test))[0]

		cond_stage_train = self.cond_stage_data[indices_train]
		cond_stage_test = self.cond_stage_data[indices_test]

		num_train, num_test = len(indices_train), len(indices_test)


		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
									 cond_stage_train),
			batch_size=batch_size,
			shuffle=False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
									 cond_stage_test),
			batch_size=batch_size,
			shuffle=False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_train[indices],
																					 give_mean=False)).float().to(device)
				batch_cond = self.encode_con(batch_cond_x.float().to(device))

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_test[indices],
																					 give_mean=False)).float().to(
					device)
				batch_cond = self.encode_con(batch_cond_x.float().to(device))

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= num_train
			test_loss /= num_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)


		
	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite: 
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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


class Net2NetFlow_scVIGenoPerLibFlow(nn.Module):
	"""
	cINN module connecting GenotypeVAE latent space to scVI latent space
	with pre-defined onehot-encoding library for perturbations
	"""
	def __init__(self,
				 configured_flow,
				 cond_stage_data,
				 perturbToOnehotLib,
				 oneHotData,
				 model_con,
				 scvi_model,
				 model_to_use = None,
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.cond_stage_data = cond_stage_data
		self.oneHotData = oneHotData
		self.perturbToOnehotLib = perturbToOnehotLib
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.model_con = model_con
		self.scvi_model = scvi_model

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime


	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	@torch.no_grad()
	def extractOneHot(self, batch_perturb):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[self.cond_stage_data[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])

	@torch.no_grad()
	def extractOneHotUnseen(self, batch_perturb, cond_stage_data_unseen):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[cond_stage_data_unseen[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])

	##########################
	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr = lr,
							   betas = (0.5, 0.9),
							   amsgrad = True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr = lr,
							   #betas=(0.5, 0.9),
							   amsgrad = False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt


	def train_evaluateUnseenPer(self, anndata_unseen, cond_stage_data_unseen, path_save = None,
								n_epochs = 400, batch_size = 128, lr = 4.5e-6,
								train_ratio = 0.8, seed = 42, start_epoch = 1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		begin = time.time()
		n = self.cond_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train
		n_unseen = len(cond_stage_data_unseen)

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))
		cond_stage_unseen = torch.tensor(np.array(list(range(len(cond_stage_data_unseen)))).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
									 cond_stage_train),
			batch_size = batch_size,
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
									 cond_stage_test),
			batch_size = batch_size,
			shuffle = False)

		valid_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_unseen,
									 cond_stage_unseen),
			batch_size = batch_size,
			shuffle = False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		unseen_loss_list = []

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0
			unseen_loss = 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):

				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)


				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x.float().to(device))

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)

				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			for batch_idx, (_, batch_cond_x, indices) in enumerate(valid_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(adata = anndata_unseen[indices.cpu().detach().numpy(), :].copy(),
																					 give_mean = False)).float().to(device)
				batch_cond_x = self.extractOneHotUnseen(batch_cond_x, cond_stage_data_unseen).float().to(device)
				batch_cond = self.encode_con(batch_cond_x).to(device)
				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				unseen_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
			unseen_loss /= n_unseen

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)
			unseen_loss_list.append(unseen_loss)

		if path_save is not None:
			np.save(os.path.join(path_save, "unseen_loss.npy"), np.array(unseen_loss_list))

		self.training_time += (time.time() - begin)


	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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




class Net2NetFlow_scVIGenoStatesFlow(nn.Module):
	"""
	cINN module connecting GenotypeVAE latent space and cell 
	state covariates to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 cond_stage_data,
				 model_con,
				 scvi_model,
				 cell_type_onehot,
				 dose_onehot,
				 model_to_use=None,
				 ignore_keys=[],
				 first_stage_key="cell",
				 cond_stage_key="cell",
				 interpolate_cond_size=-1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.model_con = model_con
		self.scvi_model = scvi_model
		self.cell_type_onehot = cell_type_onehot
		self.dose_onehot = dose_onehot

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime

	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	##########################
	def shared_step(self, batch_first, batch_cond, batch_idx, split="train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first  # self.get_input(self.first_stage_key, batch)
		c = batch_cond  # self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split=split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr=lr,
							   # betas=(0.5, 0.9),
							   amsgrad=False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		begin = time.time()

		n = self.cond_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		dose_cell_onehot = np.concatenate((self.dose_onehot, self.cell_type_onehot), axis=1)
		first_stage_train, first_stage_test = dose_cell_onehot[indices_train], dose_cell_onehot[indices_test]

		cond_stage_train = self.cond_stage_data[indices_train]
		cond_stage_test = self.cond_stage_data[indices_test]

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_train,
									 cond_stage_train),
			batch_size=batch_size,
			shuffle=False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_test,
									 cond_stage_test),
			batch_size=batch_size,
			shuffle=False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_doseCell_x, batch_cond_x, indices) in enumerate(train_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_train[indices],
																					 give_mean=False)).float().to(
					device)
				batch_cond_p = self.encode_con(batch_cond_x.float().to(device))
				batch_cond = torch.cat((batch_cond_p, batch_doseCell_x.to(device)), 1)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (batch_doseCell_x, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_test[indices],
																					 give_mean=False)).float().to(
					device)
				batch_cond_p = self.encode_con(batch_cond_x.float().to(device))
				batch_cond = torch.cat((batch_cond_p, batch_doseCell_x.to(device)), 1)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok=overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)

	def load(self, dir_path: str, use_cuda: bool = False, save_use_cuda: bool = False):
		use_cuda = use_cuda and torch.cuda.is_available()
		device = torch.device('cpu') if use_cuda is False else torch.device('cuda')

		model_save_path = os.path.join(dir_path, "model_params.pt")

		if use_cuda and save_use_cuda:
			self.load_state_dict(torch.load(model_save_path))
			self.to(device)
		elif use_cuda and save_use_cuda is False:
			self.load_state_dict(torch.load(model_save_path, map_location="cuda:0"))
			self.to(device)
		else:
			self.load_state_dict(torch.load(model_save_path, map_location=device))


class Net2NetFlow_TFVAENonStdFlow(nn.Module):
	"""
	cINN module connecting GenotypeVAE latent space to VAE latent space
	with pre-defined onehot-encoding library for perturbations
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data,
				 cond_stage_data,
				 perturbToOnehotLib,
				 oneHotData,
				 model_con,
				 sess, enc_ph, z_gen_data_v, is_training,
				 model_to_use=None,
				 ignore_keys=[],
				 first_stage_key="cell",
				 cond_stage_key="cell",
				 interpolate_cond_size=-1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.first_stage_data = first_stage_data
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.perturbToOnehotLib = perturbToOnehotLib
		self.oneHotData = oneHotData
		self.model_con = model_con
		self.sess = sess
		self.enc_ph = enc_ph
		self.z_gen_data_v = z_gen_data_v
		self.is_training = is_training

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime

	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	##########################

	def get_input(self, key, batch, is_conditioning=False):
		x = batch
		if key in ["caption", "text"]:
			x = list(x[0])
		elif key in ["class"]:
			pass
		else:
			if len(x.shape) == 2:
				x = x[..., None, None]
			elif len(x.shape) == 3:
				x = x[..., None]
			x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
			if is_conditioning:
				if self.interpolate_cond_size > -1:
					x = F.interpolate(x, size=(self.interpolate_cond_size, self.interpolate_cond_size))
		return x

	def shared_step(self, batch_first, batch_cond, batch_idx, split="train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first  # self.get_input(self.first_stage_key, batch)
		c = batch_cond  # self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split=split)
		return loss, log_dict

	def configure_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	@torch.no_grad()
	def encode(self, x_torch):
		x_np = x_torch.detach().cpu().numpy()
		feed_dict = {self.enc_ph: x_np, self.is_training: False}
		z = self.sess.run(self.z_gen_data_v, feed_dict=feed_dict)
		z_torch = torch.tensor(z).float()
		return z_torch.unsqueeze(-1).unsqueeze(-1)

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	@torch.no_grad()
	def extractOneHot(self, batch_perturb):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[self.cond_stage_data[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])

	@torch.no_grad()
	def extractOneHotUnseen(self, batch_perturb, cond_stage_data_unseen):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[cond_stage_data_unseen[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])

	def train(self, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()

		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]

		# cond_stage_train = torch.tensor(np.array([cond_stage_train]).reshape(-1))
		# cond_stage_test = torch.tensor(np.array([cond_stage_test]).reshape(-1))

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_train,
						  cond_stage_train),
			batch_size=batch_size,
			shuffle=True)

		test_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_test,
						  cond_stage_test),
			batch_size=batch_size,
			shuffle=True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(train_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)


				batch_cond = self.encode_con(batch_cond_x).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(test_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)

				batch_cond = self.encode_con(batch_cond_x).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

	def train_evaluateUnseenPer(self, data_unseen, cond_stage_data_unseen, path_save = None, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()

		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train
		n_unseen = len(cond_stage_data_unseen)

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]

		# cond_stage_train = torch.tensor(np.array([cond_stage_train]).reshape(-1))
		# cond_stage_test = torch.tensor(np.array([cond_stage_test]).reshape(-1))

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))
		cond_stage_unseen = torch.tensor(np.array(list(range(len(cond_stage_data_unseen)))).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_train,
						  cond_stage_train),
			batch_size=batch_size,
			shuffle=True)

		test_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_test,
						  cond_stage_test),
			batch_size=batch_size,
			shuffle=True)

		valid_loader = torch.utils.data.DataLoader(
			ConcatDataset(data_unseen,
						  cond_stage_unseen),
			batch_size = batch_size,
			shuffle = True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		unseen_loss_list = []

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0
			unseen_loss = 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(train_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)

				batch_cond = self.encode_con(batch_cond_x).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(test_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)

				batch_cond = self.encode_con(batch_cond_x).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(valid_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHotUnseen(batch_cond_x, cond_stage_data_unseen).float().to(device)

				batch_cond = self.encode_con(batch_cond_x).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				unseen_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
			unseen_loss /= n_unseen

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)
			unseen_loss_list.append(unseen_loss)

		if path_save is not None:
			np.save(os.path.join(path_save, "unseen_loss.npy"), np.array(unseen_loss_list))

		self.training_time += (time.time() - begin)


	def train_trainTestDiffPer(self, perturb_arr, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()

		perturb_unique = np.unique(perturb_arr)
		n = len(perturb_unique)
		n_train = int(n * 0.8)
		n_test = n - n_train

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_per_test, indices_per_train = permutation[:n_test], permutation[n_test:]
		perturb_train, perturb_test = perturb_unique[indices_per_train], perturb_unique[indices_per_test]

		indices_train = np.where(np.in1d(perturb_arr, perturb_train))[0]
		indices_test = np.where(np.in1d(perturb_arr, perturb_test))[0]

		first_stage_train = self.first_stage_data[indices_train]
		first_stage_test = self.first_stage_data[indices_test]

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))

		num_train, num_test = len(indices_train), len(indices_test)

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_train,
						  cond_stage_train),
			batch_size=batch_size,
			shuffle=True)

		test_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_test,
						  cond_stage_test),
			batch_size=batch_size,
			shuffle=True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(train_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)

				batch_cond = self.encode_con(batch_cond_x).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(test_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).float().to(device)

				batch_cond = self.encode_con(batch_cond_x).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= num_train
			test_loss /= num_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok=overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)

	def load(self, dir_path: str, use_cuda: bool = False, save_use_cuda: bool = False):
		use_cuda = use_cuda and torch.cuda.is_available()
		device = torch.device('cpu') if use_cuda is False else torch.device('cuda')

		model_save_path = os.path.join(dir_path, "model_params.pt")

		if use_cuda and save_use_cuda:
			self.load_state_dict(torch.load(model_save_path))
			self.to(device)
		elif use_cuda and save_use_cuda is False:
			self.load_state_dict(torch.load(model_save_path, map_location="cuda:0"))
			self.to(device)
		else:
			self.load_state_dict(torch.load(model_save_path, map_location=device))


class Net2NetFlow_TFVAEFixFlow(nn.Module):
	"""
	cINN module connecting ESM latent space and Gaussian noise of pre-defined standard deviations
	to VAE latent space
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data,
				 cond_stage_data,
				 perturbToEmbedLib,
				 embedData,
				 sess, enc_ph, z_gen_data_v, is_training,
				 model_to_use=None,
				 ignore_keys=[],
				 first_stage_key="cell",
				 cond_stage_key="cell",
				 interpolate_cond_size=-1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.first_stage_data = first_stage_data
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.perturbToEmbedLib = perturbToEmbedLib
		self.embedData = embedData
		self.sess = sess
		self.enc_ph = enc_ph
		self.z_gen_data_v = z_gen_data_v
		self.is_training = is_training

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime

	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	def shared_step(self, batch_first, batch_cond, batch_idx, split="train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first  # self.get_input(self.first_stage_key, batch)
		c = batch_cond  # self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split=split)
		return loss, log_dict

	def configure_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	@torch.no_grad()
	def encode(self, x_torch):
		x_np = x_torch.detach().cpu().numpy()
		feed_dict = {self.enc_ph: x_np, self.is_training: False}
		z = self.sess.run(self.z_gen_data_v, feed_dict=feed_dict)
		z_torch = torch.tensor(z).float()
		return z_torch.unsqueeze(-1).unsqueeze(-1)

	@torch.no_grad()
	def encode_con(self, z_con, device, sigma_epsilon = 0.001):
		noise = torch.normal(mean = 0, std = sigma_epsilon, size = (z_con.shape[0], z_con.shape[1])).to(device)
		return noise + z_con

	@torch.no_grad()
	def extractEmbed(self, batch_perturb):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToEmbedLib[self.cond_stage_data[batch_perturb[i].item()]])
		return torch.tensor(self.embedData[indx])

	@torch.no_grad()
	def extractEmbedUnseen(self, batch_perturb, cond_stage_data_unseen):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToEmbedLib[cond_stage_data_unseen[batch_perturb[i].item()]])
		return torch.tensor(self.embedData[indx])

	def train_evaluateUnseenPer(self, data_unseen, cond_stage_data_unseen, path_save = None, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1, sigma_epsilon = 0.001):
		"""
		train the net2net model, with train and test validation
		"""

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()

		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train
		n_unseen = len(cond_stage_data_unseen)

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))
		cond_stage_unseen = torch.tensor(np.array(list(range(len(cond_stage_data_unseen)))).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_train,
						  cond_stage_train),
			batch_size=batch_size,
			shuffle=True)

		test_loader = torch.utils.data.DataLoader(
			ConcatDataset(first_stage_test,
						  cond_stage_test),
			batch_size=batch_size,
			shuffle=True)

		valid_loader = torch.utils.data.DataLoader(
			ConcatDataset(data_unseen,
						  cond_stage_unseen),
			batch_size = batch_size,
			shuffle = True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		unseen_loss_list = []

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0
			unseen_loss = 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(train_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)

				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(test_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)

				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			for batch_idx, (batch_first_x, batch_cond_x) in enumerate(valid_loader):
				# batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices, give_mean = False)).float().to(device)
				batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractEmbedUnseen(batch_cond_x, cond_stage_data_unseen).float().to(device)

				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				unseen_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
			unseen_loss /= n_unseen

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)
			unseen_loss_list.append(unseen_loss)

		if path_save is not None:
			np.save(os.path.join(path_save, "unseen_loss.npy"), np.array(unseen_loss_list))

		self.training_time += (time.time() - begin)

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok=overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)

	def load(self, dir_path: str, use_cuda: bool = False, save_use_cuda: bool = False):
		use_cuda = use_cuda and torch.cuda.is_available()
		device = torch.device('cpu') if use_cuda is False else torch.device('cuda')

		model_save_path = os.path.join(dir_path, "model_params.pt")

		if use_cuda and save_use_cuda:
			self.load_state_dict(torch.load(model_save_path))
			self.to(device)
		elif use_cuda and save_use_cuda is False:
			self.load_state_dict(torch.load(model_save_path, map_location="cuda:0"))
			self.to(device)
		else:
			self.load_state_dict(torch.load(model_save_path, map_location=device))

class Net2NetFlow_scVIFixFlow(nn.Module):
	"""
	cINN module connecting ESM latent space and Gaussian noise of pre-defined standard deviations
	to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 cond_stage_data,
				 perturbToEmbedLib,
				 embedData,
				 scvi_model,
				 model_to_use = None,
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.perturbToEmbedLib = perturbToEmbedLib
		self.embedData = embedData
		self.scvi_model = scvi_model

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime


	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, z_con, device, sigma_epsilon = 0.001):
		noise = torch.normal(mean = 0, std = sigma_epsilon, size = (z_con.shape[0], z_con.shape[1])).to(device)
		return noise + z_con

	@torch.no_grad()
	def extractEmbed(self, batch_perturb):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToEmbedLib[self.cond_stage_data[batch_perturb[i].item()]])
		return torch.tensor(self.embedData[indx])

	@torch.no_grad()
	def extractEmbedUnseen(self, batch_perturb, cond_stage_data_unseen):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToEmbedLib[cond_stage_data_unseen[batch_perturb[i].item()]])
		return torch.tensor(self.embedData[indx])

	##########################
	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr = lr,
							   betas = (0.5, 0.9),
							   amsgrad = True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr = lr,
							   #betas=(0.5, 0.9),
							   amsgrad = False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self,
								n_epochs = 400, batch_size = 128, lr = 4.5e-6,
								train_ratio = 0.8, seed = 42, start_epoch = 1, sigma_epsilon = 0.001):
		"""
		train the net2net model, with train and test validation
		"""

		begin = time.time()
		n = self.cond_stage_data.shape[0]
		n_train = int(n * train_ratio)
		n_test = n - n_train


		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]


		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))



		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
									 cond_stage_train),
			batch_size = batch_size,
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
									 cond_stage_test),
			batch_size = batch_size,
			shuffle = False)



		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0


			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):
				#print(len(indices))    
				if len(indices) == 1:
					continue
				#print(indices_train[indices])
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)

				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				if len(indices) == 1:
					continue
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)

				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]



			train_loss /= n_train
			test_loss /= n_test


			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)



		self.training_time += (time.time() - begin)
        
        
        
        
	def train_evaluateUnseenPer(self, anndata_unseen, cond_stage_data_unseen, path_save = None,
								n_epochs = 400, batch_size = 128, lr = 4.5e-6,
								train_ratio = 0.8, seed = 42, start_epoch = 1, sigma_epsilon = 0.001):
		"""
		train the net2net model, with train and test validation
		"""

		begin = time.time()
		n = self.cond_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train
		n_unseen = len(cond_stage_data_unseen)

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]


		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))
		cond_stage_unseen = torch.tensor(np.array(list(range(len(cond_stage_data_unseen)))).reshape(-1))


		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
									 cond_stage_train),
			batch_size = batch_size,
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
									 cond_stage_test),
			batch_size = batch_size,
			shuffle = False)

		valid_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_unseen,
									 cond_stage_unseen),
			batch_size = batch_size,
			shuffle = False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		unseen_loss_list = []

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0
			unseen_loss = 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):

				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)

				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)

				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			for batch_idx, (_, batch_cond_x, indices) in enumerate(valid_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(adata = anndata_unseen[indices.cpu().detach().numpy(), :].copy(),
																					 give_mean = False)).float().to(device)
				batch_cond_x = self.extractEmbedUnseen(batch_cond_x, cond_stage_data_unseen).float().to(device)
				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				unseen_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
			unseen_loss /= n_unseen

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)
			unseen_loss_list.append(unseen_loss)

		if path_save is not None:
			np.save(os.path.join(path_save, "unseen_loss.npy"), np.array(unseen_loss_list))

		self.training_time += (time.time() - begin)


	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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
            

	#New TFVAE with covariates
class Net2NetFlow_TFVAE_Covariate_Flow(nn.Module):
	"""
	cINN module connecting standardized ChemicalVAE latent space to VAE/scvi latent space.Hgraph also implemented
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data, 
				 cond_stage_data, 
				 perturbToOnehotLib,
				 oneHotData, 
				 model_con, 
				 std_model, 
				 sess = None, enc_ph = None, z_gen_data_v = None, is_training = None,
				 covariates = None,
				 scvi_model = None,
				 model_to_use = None,
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.first_stage_data = first_stage_data
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.perturbToOnehotLib = perturbToOnehotLib 
		self.oneHotData = oneHotData
		self.model_con = model_con
		self.std_model = std_model 
		self.sess = sess
		self.enc_ph = enc_ph
		self.z_gen_data_v = z_gen_data_v
		self.is_training = is_training
		self.covariates = covariates
		self.scvi_model = scvi_model

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use 
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime

	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec
	##########################

	def get_input(self, key, batch, is_conditioning = False):
		x = batch
		if key in ["caption", "text"]:
			x = list(x[0])
		elif key in ["class"]:
			pass
		else:
			if len(x.shape) == 2:
				x = x[..., None, None]
			elif len(x.shape) == 3:
				x = x[..., None]
			x = x.permute(0, 3, 1, 2).to(memory_format = torch.contiguous_format)
			if is_conditioning:
				if self.interpolate_cond_size > -1:
					x = F.interpolate(x, size=(self.interpolate_cond_size, self.interpolate_cond_size))
		return x

	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train",reverse = False):
		"""
		compute the loss value in a batch 
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		if reverse:
			x = x.squeeze(-1).squeeze(-1)
			c = c.unsqueeze(-1).unsqueeze(-1)
			zz, logdet = self(c, x)
		else:
			zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict

	def configure_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	@torch.no_grad()
	def encode(self, x_torch):
		x_np = x_torch.detach().cpu().numpy()
		feed_dict = {self.enc_ph: x_np, self.is_training: False}
		z = self.sess.run(self.z_gen_data_v, feed_dict = feed_dict)
		z_torch = torch.tensor(z).float()
		return z_torch.unsqueeze(-1).unsqueeze(-1)

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z



	@torch.no_grad()
	def extractOneHot(self, batch_perturb):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[self.cond_stage_data[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])
    


	@torch.no_grad()
	def extractOneHotUnseen(self, batch_perturb, cond_stage_data_unseen):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToOnehotLib[cond_stage_data_unseen[batch_perturb[i].item()]])
		return torch.tensor(self.oneHotData[indx])
    


	def train_cinn_hgraph(self, n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1,reverse = False, cov_concat = "Pert",auto_save = 0, auto_save_path = "./", perturb = False):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		if self.scvi_model is None:
			scvi_flag = False
		else:
			scvi_model = self.scvi_model           
			scvi_flag = True        

		begin = time.time()


		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train 

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]
        
		#dose_cell_onehot = np.concatenate((self.dose_onehot, self.cell_type_onehot), axis = 1)
        
        
		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]
        
		#dose_cell_train, dose_cell_test = dose_cell_onehot[indices_train], dose_cell_onehot[indices_test]

		# cond_stage_train = torch.tensor(np.array([cond_stage_train]).reshape(-1))
		# cond_stage_test = torch.tensor(np.array([cond_stage_test]).reshape(-1))

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))      
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_train, 
				cond_stage_train),
			batch_size = batch_size, 
			shuffle = True)


		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_test, 
				cond_stage_test),
			batch_size = batch_size, 
			shuffle =  True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0             
			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(train_loader):
				if scvi_flag:                
					batch_first = torch.tensor(scvi_model.get_latent_representation(indices = indices_train[indices_cov_use], give_mean = False)).float().to(device).unsqueeze(-1).unsqueeze(-1)                    
				else:                    
					batch_first = self.encode(batch_first_x).to(device)
				if not perturb:
					batch_cond_x = self.extractOneHot(batch_cond_x).to(device)                 
				else:
					batch_cond_x = torch.tensor(self.cond_stage_data[batch_cond_x]).to(device).float()                 
				batch_cov_x = torch.from_numpy(self.covariates[indices_train[indices_cov_use]])       
				if cov_concat == "Pert":
					batch_cond = torch.cat((batch_cond_x, batch_cov_x.to(device)), 1)
				else:
					batch_first =torch.cat((batch_first.squeeze(-1).squeeze(-1), batch_cov_x.to(device)), 1)
					batch_cond = batch_cond_x
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx, reverse = reverse)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]


			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(test_loader):
				if scvi_flag:                
					batch_first = torch.tensor(scvi_model.get_latent_representation(indices = indices_test[indices_cov_use], give_mean = False)).float().to(device).unsqueeze(-1).unsqueeze(-1)
				else:                    
					batch_first = self.encode(batch_first_x).to(device)
				if not perturb:
					batch_cond_x = self.extractOneHot(batch_cond_x).to(device)                 
				else:
					batch_cond_x = torch.tensor(self.cond_stage_data[batch_cond_x]).to(device).float()
				batch_cov_x = torch.from_numpy(self.covariates[indices_test[indices_cov_use]]).float()
				if cov_concat == "Pert":
					batch_cond = torch.cat((batch_cond_x, batch_cov_x.to(device)), 1)
# batch_first is actually always the coresponding to the cell states representations.
				else:
					batch_first = torch.cat((batch_first.squeeze(-1).squeeze(-1), batch_cov_x.to(device)), 1) 
					batch_cond = batch_cond_x                
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx, reverse = reverse)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
            
			if auto_save and epoch % auto_save == 0 and epoch < n_epochs:         
				auto_save_path_model = os.path.join(auto_save_path,"model_params" + str(epoch) + ".pt")
				if not os.path.exists(auto_save_path): 
					os.makedirs(auto_save_path, exist_ok = False)
				torch.save(self.state_dict(), auto_save_path_model)
                
			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)



		self.training_time += (time.time() - begin)
        
        

   
        
        

	def train_cinn(self, n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1, auto_save = 0, auto_save_path = "./"):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()
        
		if self.scvi_model is None:
			scvi_flag = False
		else:
			scvi_model = self.scvi_model         
			scvi_flag = True        

            
		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]


		begin = time.time()


		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train 

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]
        
		#dose_cell_onehot = np.concatenate((self.dose_onehot, self.cell_type_onehot), axis = 1)
        
        
		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]
        
		#dose_cell_train, dose_cell_test = dose_cell_onehot[indices_train], dose_cell_onehot[indices_test]

		# cond_stage_train = torch.tensor(np.array([cond_stage_train]).reshape(-1))
		# cond_stage_test = torch.tensor(np.array([cond_stage_test]).reshape(-1))

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))

		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_train, 
				cond_stage_train),
			batch_size = batch_size, 
			shuffle = True)


		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_test, 
				cond_stage_test),
			batch_size = batch_size, 
			shuffle =  True)

		optimizer = self.configure_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(train_loader):
				if scvi_flag:                
					batch_first = torch.tensor(scvi_model.get_latent_representation(indices = indices_train[indices_cov_use], give_mean = False)).float().to(device).unsqueeze(-1).unsqueeze(-1)
                  
				else:                    
					batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)
				batch_cov_x = torch.from_numpy(self.covariates[indices_train[indices_cov_use]]).float()
				batch_cond_p = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)
				batch_cond = torch.cat((batch_cond_p, batch_cov_x.to(device)), 1)

				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]


			self.flow.eval()
			for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(test_loader):
				if scvi_flag:                
					batch_first = torch.tensor(scvi_model.get_latent_representation(indices = indices_test[indices_cov_use], give_mean = False)).float().to(device).unsqueeze(-1).unsqueeze(-1)
				else:                    
					batch_first = self.encode(batch_first_x).to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)
				batch_cov_x = torch.from_numpy(self.covariates[indices_test[indices_cov_use]]).float()
				batch_cond_p = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)
				batch_cond = torch.cat((batch_cond_p, batch_cov_x.to(device)), 1)
                
				loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test
            
			if auto_save and epoch % auto_save == 0 and epoch < n_epochs:         
				auto_save_path_model = os.path.join(auto_save_path,"model_params" + str(epoch) + ".pt")
				if not os.path.exists(auto_save_path): 
					os.makedirs(auto_save_path, exist_ok = False)
				torch.save(self.state_dict(), auto_save_path_model)
                
			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)



		self.training_time += (time.time() - begin)
        

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite: 
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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

class Net2NetFlow_scVIGenoFlow_GIlayer(nn.Module):
	"""
	cINN module connecting GenotypeVAE latent space to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 cond_stage_data,
				 model_con, 
				 scvi_model,
				 perturbToOnehotLib,
				 geneToOnehotLib,
				 oneHotData,     
				 model_to_use = None, 
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.model_con = model_con
		self.scvi_model = scvi_model
		self.perturbToOnehotLib = perturbToOnehotLib
		self.geneToOnehotLib = geneToOnehotLib        
		self.oneHotData = oneHotData
		#self.temp_state_list = []
		#self.temp_params_list = []
		self.count_dict = {}
		# GI layer
		self.linear_1 = nn.Linear(20, 15)
		self.linear_2 = nn.Linear(15, 12)
		self.linear_3 = nn.Linear(12, 10)
		self.bn1 = nn.BatchNorm1d(15)
		self.bn2 = nn.BatchNorm1d(12)
		self.dropout1 = nn.Dropout(p = 0.2)
		self.dropout2 = nn.Dropout(p = 0.2)       
		self.leaky = nn.LeakyReLU(negative_slope = 0.2)
        
        
	def GI_model(self, x):
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
		return self.linear_3(x)

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use 
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime


	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	@torch.no_grad()
	def extractOneHot_scvi(self, batch_perturb):
		indx = []
		for i in range(len(batch_perturb)):
			indx.append(self.perturbToOnehotLib[batch_perturb[i]])
		return torch.tensor(self.oneHotData[indx])
    

	def encode_GI(self, batch_perturb):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		indx = []
		GI_embedding = []
		emb_mat = []
		for i in range(len(batch_perturb)):
			#print(batch_perturb[i])
			pert = batch_perturb[i].split("/")
			ge1 = pert[0]
			ge2 = pert[1]
			if ge2  == "ctrl":
				continue
			else:
				indx.append(i)      
				emb1 = self.geneToOnehotLib[ge1]
				emb2 = self.geneToOnehotLib[ge2]               
				emb = np.concatenate([emb1,emb2])
				emb_mat.append(emb.reshape(1,len(emb)))
		if len(emb_mat) != 0:
			emb_mat = np.concatenate(emb_mat)
			#print(emb_mat.shape)           
			GI_embedding.append(self.GI_model(torch.tensor(emb_mat).float().to(device)))
		if len(indx) == 0:
			return indx, GI_embedding
		else:
			return indx, torch.cat(GI_embedding)
    

	##########################
	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train"):
		"""
		compute the loss value in a batch 
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr = lr,
							   betas = (0.5, 0.9),
							   amsgrad = True)
		return opt
    
	def configure_GIflow_optimizers(self, lr):
		param_GI = []
		GINN = ["linear_1.weight","linear_1.bias","linear_2.weight","linear_2.bias","linear_3.weight","linear_3.bias","bn1.weight"
                ,"bn1.bias","bn2.weight","bn2.bias"]
		for name, param in self.named_parameters():
			if name in GINN:
				param_GI.append(param)
				#print(name)

		params = list(self.flow.parameters()) + param_GI
		opt = torch.optim.Adam(params,
							   lr = lr,
							   betas = (0.5, 0.9),
							   amsgrad = True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr = lr,
							   #betas=(0.5, 0.9),
							   amsgrad = False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self, n_epochs = 400, batch_size = 128, lr = 4.5e-6, train_ratio = 0.8, seed = 42, start_epoch = 1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		begin = time.time()


		n = self.cond_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train 

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		cond_stage_train = self.cond_stage_data[indices_train]
		cond_stage_test = self.cond_stage_data[indices_test]


		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train, 
				cond_stage_train),
			batch_size = batch_size, 
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
				cond_stage_test),
			batch_size = batch_size, 
			shuffle = False)

		optimizer = self.configure_GIflow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(train_loader):

				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices], give_mean = False)).float().to(device)
				#print(batch_cond_x)          
				batch_cond_x_onehot = self.extractOneHot_scvi(batch_cond_x).to(device)
                
				batch_cond = self.encode_con(batch_cond_x_onehot.float().to(device))
				GI_indx, batch_cond_GI = self.encode_GI(batch_cond_x)
				#print(batch_cond_GI.size())
				#print(batch_cond[GI_indx[0],:])
				if len(GI_indx) != 0:
					batch_cond[GI_indx,:] = batch_cond_GI
				#print(batch_cond[GI_indx[0],:])

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]
				#self.temp_state_list.append(self.state_dict())
				#self.temp_params_list.append(self.named_parameters())
			#if epoch == 1:
				#for name, param in self.named_parameters():
					#if 'weight' in name and param.grad != None:
						#print(name)
						#self.count_dict[name] = torch.zeros(param.grad.shape)
			#else:    
				#for name, param in self.named_parameters():
					#if 'weight' in name and param.grad != None:
						#temp = torch.zeros(param.grad.shape).cpu()
						#temp[param.grad.cpu() != 0] += 1
						#self.count_dict[name] += temp


			self.flow.eval()
			for batch_idx, (_, batch_cond_x, indices) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices], give_mean = False)).float().to(device)
				#print(batch_cond_x)          
				batch_cond_x_onehot = self.extractOneHot_scvi(batch_cond_x).to(device)
                
				batch_cond = self.encode_con(batch_cond_x_onehot.float().to(device))
				GI_indx, batch_cond_GI = self.encode_GI(batch_cond_x)
				#print(batch_cond_GI.size())
				#print(batch_cond[GI_indx[0],:])
				if len(GI_indx) != 0:
					batch_cond[GI_indx,:] = batch_cond_GI
				#print(batch_cond[GI_indx[0],:])


				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)



	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite: 
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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
            
            
class Net2NetFlow_scVIChemFlow(nn.Module):
	"""
	cINN module connecting standardized ChemicalVAE latent space and cell state
	covariates to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data,
				 cond_stage_data,
				 model_con,
				 scvi_model,
				 std_model,
				 covariates = None,
				 model_to_use=None,
				 ignore_keys=[],
				 first_stage_key="cell",
				 cond_stage_key="cell",
				 interpolate_cond_size=-1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.first_stage_data = first_stage_data
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.model_con = model_con
		self.scvi_model = scvi_model
		self.std_model = std_model
		self.covariates = covariates

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime

	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, x_con):
		_, _, _, z = self.model_con(x_con)
		return z

	##########################

	def shared_step(self, batch_first, batch_cond, batch_idx, split="train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first  # self.get_input(self.first_stage_key, batch)
		c = batch_cond  # self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split=split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr=lr,
							   # betas=(0.5, 0.9),
							   amsgrad=False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self, n_epochs=400, batch_size=128, lr=4.5e-6, train_ratio=0.8, seed=42, start_epoch=1):
		"""
		train the net2net model, with train and test validation
		"""
		self.model_con.eval()

		assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

		begin = time.time()

		n = self.first_stage_data.shape[0]
		n_train = int(n * 0.8)
		n_test = n - n_train

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed=seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		#dose_cell_onehot = np.concatenate((self.dose_onehot, self.cell_type_onehot), axis = 1)

		first_stage_train, cond_stage_train = self.first_stage_data[indices_train], self.cond_stage_data[indices_train]
		first_stage_test, cond_stage_test = self.first_stage_data[indices_test], self.cond_stage_data[indices_test]

		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))
        
        
		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_train,
									 cond_stage_train),
			batch_size=batch_size,
			shuffle=False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(first_stage_test,
									 cond_stage_test),
			batch_size=batch_size,
			shuffle=False)

		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0

			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(train_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_train[indices_cov_use],
																					 give_mean=False)).float().to(device)
				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)
				batch_cov_x = torch.from_numpy(self.covariates[indices_train[indices_cov_use]]).float()
				batch_cond_p = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)
				batch_cond = torch.cat((batch_cond_p, batch_cov_x.to(device)), 1)


				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for batch_idx, (atch_first_x, batch_cond_x, indices_cov_use) in enumerate(test_loader):
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices=indices_test[indices_cov_use],
																					 give_mean=False)).float().to(
					device)
				# batch_cond_x = batch_cond_x.float().to(device)
				# batch_cond_x = self.std_model.standardize_z_torch(batch_cond_x.float()).to(device)
				# batch_cond = self.encode_con(batch_cond_x).to(device)

				batch_cond_x = self.extractOneHot(batch_cond_x).to(device)
				batch_cov_x = torch.from_numpy(self.covariates[indices_train[indices_cov_use]]).float()
				batch_cond_p = self.std_model.standardize_z_torch(self.encode_con(batch_cond_x).to(device)).to(device)
				batch_cond = torch.cat((batch_cond_p, batch_cov_x.to(device)), 1)

				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]

			train_loss /= n_train
			test_loss /= n_test

			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)

		self.training_time += (time.time() - begin)

	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok=overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)

	def load(self, dir_path: str, use_cuda: bool = False, save_use_cuda: bool = False):
		use_cuda = use_cuda and torch.cuda.is_available()
		device = torch.device('cpu') if use_cuda is False else torch.device('cuda')

		model_save_path = os.path.join(dir_path, "model_params.pt")

		if use_cuda and save_use_cuda:
			self.load_state_dict(torch.load(model_save_path))
			self.to(device)
		elif use_cuda and save_use_cuda is False:
			self.load_state_dict(torch.load(model_save_path, map_location="cuda:0"))
			self.to(device)
		else:
			self.load_state_dict(torch.load(model_save_path, map_location=device))

class Net2NetFlow_scVIFix_Covariate_Flow(nn.Module):
	"""
	cINN module connecting ESM latent space and Gaussian noise of pre-defined standard deviations
	to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 cond_stage_data,
				 perturbToEmbedLib,
				 embedData,

				 scvi_model,
				 covariates = None,
				 model_to_use = None,
				 ignore_keys = [],
				 first_stage_key = "cell",
				 cond_stage_key = "cell",
				 interpolate_cond_size = -1
				 ):
		super().__init__()
		self.flow = configured_flow
		self.loss = NLL()
		self.cond_stage_data = cond_stage_data
		self.first_stage_key = first_stage_key
		self.cond_stage_key = cond_stage_key
		self.interpolate_cond_size = interpolate_cond_size
		self.training_time = 0
		self.train_loss = []
		self.test_loss = []
		self.perturbToEmbedLib = perturbToEmbedLib
		self.embedData = embedData
		self.covariates = covariates        
		self.scvi_model = scvi_model

	def forward(self, x, c):
		zz, logdet = self.flow(x, c)
		return zz, logdet

	@torch.no_grad()
	def sample_conditional(self, c):
		"""
		sample zz for c to use
		"""
		z = self.flow.sample(c)
		return z

	@torch.no_grad()
	def generate_zprime(self, x, c, cprime):
		"""
		generate new x from x (2 dimensional) and c
		"""
		zz, _ = self.flow(x, c)
		zprime = self.flow.reverse(zz, cprime)
		return zprime


	@torch.no_grad()
	def generate_zrec(self, x, c):

		zz, _ = self.flow(x, c)
		zrec = self.flow.reverse(zz, c)
		return zrec

	@torch.no_grad()
	def encode_con(self, z_con, device, sigma_epsilon = 0.001):
		noise = torch.normal(mean = 0, std = sigma_epsilon, size = (z_con.shape[0], z_con.shape[1])).to(device)
		return noise + z_con

	@torch.no_grad()
	def extractEmbed(self, batch_perturb):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToEmbedLib[self.cond_stage_data[batch_perturb[i].item()]])
		return torch.tensor(self.embedData[indx])

	@torch.no_grad()
	def extractEmbedUnseen(self, batch_perturb, cond_stage_data_unseen):
		indx = []
		for i in range(batch_perturb.shape[0]):
			indx.append(self.perturbToEmbedLib[cond_stage_data_unseen[batch_perturb[i].item()]])
		return torch.tensor(self.embedData[indx])

	##########################
	def shared_step(self, batch_first, batch_cond, batch_idx, split = "train"):
		"""
		compute the loss value in a batch
		"""
		x = batch_first #self.get_input(self.first_stage_key, batch)
		c = batch_cond #self.get_input(self.cond_stage_key, batch, is_conditioning = True)
		zz, logdet = self(x, c)
		loss, log_dict = self.loss(zz, logdet, split = split)
		return loss, log_dict

	def configure_flow_optimizers(self, lr):
		opt = torch.optim.Adam((self.flow.parameters()),
							   lr = lr,
							   betas = (0.5, 0.9),
							   amsgrad = True)
		return opt

	def configure_vae_optimizers(self, lr):

		opt = torch.optim.Adam(self.model_con.parameters(),
							   lr = lr,
							   #betas=(0.5, 0.9),
							   amsgrad = False)
		return opt

	def configure_vaeflow_optimizers(self, lr):

		params = list(self.flow.parameters()) + list(self.model_con.parameters())
		opt = torch.optim.Adam(params,
							   lr=lr,
							   betas=(0.5, 0.9),
							   amsgrad=True)
		return opt

	def train(self,
								n_epochs = 400, batch_size = 128, lr = 4.5e-6,
								train_ratio = 0.8, seed = 42, start_epoch = 1, sigma_epsilon = 0.001):
		"""
		train the net2net model, with train and test validation
		"""

		begin = time.time()
		n = self.cond_stage_data.shape[0]
		n_train = int(n * train_ratio)
		n_test = n - n_train


		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]


		cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
		cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))



		# training
		train_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_train,
									 cond_stage_train),
			batch_size = batch_size,
			shuffle = False)

		test_loader = torch.utils.data.DataLoader(
			ConcatDatasetWithIndices(cond_stage_test,
									 cond_stage_test),
			batch_size = batch_size,
			shuffle = False)



		optimizer = self.configure_flow_optimizers(lr)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


		for epoch in range(start_epoch, n_epochs + 1):
			train_loss, test_loss = 0, 0


			self.flow.train()
			for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(train_loader):
				#print(len(indices))    
				if len(indices_cov_use) == 1:
					continue
				#print(indices_train[indices])
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_train[indices_cov_use], give_mean = False)).float().to(device)
				batch_cov_x = torch.from_numpy(self.covariates[indices_train[indices_cov_use]]).float()
				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)
				batch_cond = torch.cat((batch_cond, batch_cov_x.to(device)), 1)
				#return batch_cov_x,   batch_cond_x, batch_idx,indices_cov_use, indices_train              
				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.item() * batch_first.shape[0]

			self.flow.eval()
			for  batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(test_loader):
				if len(indices_cov_use) == 1:
					continue
				batch_first = torch.tensor(self.scvi_model.get_latent_representation(indices = indices_test[indices_cov_use], give_mean = False)).float().to(device)
				batch_cov_x = torch.from_numpy(self.covariates[indices_test[indices_cov_use]]).float()
				batch_cond_x = self.extractEmbed(batch_cond_x).float().to(device)
				batch_cond = self.encode_con(batch_cond_x, device, sigma_epsilon).to(device)
				batch_cond = torch.cat((batch_cond, batch_cov_x.to(device)), 1)            
				loss, _ = self.shared_step(batch_first.unsqueeze(-1).unsqueeze(-1), batch_cond, batch_idx)
				test_loss += loss.item() * batch_first.shape[0]



			train_loss /= n_train
			test_loss /= n_test


			print(
				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
			)

			self.train_loss.append(train_loss)
			self.test_loss.append(test_loss)



		self.training_time += (time.time() - begin)
        
        
        
        



	def save(self, dir_path: str, overwrite: bool = False):
		if not os.path.exists(dir_path) or overwrite:
			os.makedirs(dir_path, exist_ok = overwrite)
		else:
			raise ValueError(
				"{} already exists, Please provide an unexisting director for saving.".format(dir_path))
		model_save_path = os.path.join(dir_path, "model_params.pt")

		torch.save(self.state_dict(), model_save_path)

		np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)


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