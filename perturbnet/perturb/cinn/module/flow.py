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

from net2net.modules.flow.loss import NLL
from net2net.ckpt_util import get_ckpt_path
from net2net.modules.util import log_txt_as_img
from net2net.modules.flow.flatflow import *

from pytorch_scvi.distributions import *
from pytorch_scvi.scvi_generate_z import *

from perturbnet.perturb.util import *
from perturbnet.perturb.data_vae.morphology.modules.image_vae import *

class Net2NetFlow_scVIChemStdFlow(nn.Module):
	"""
	cINN module connecting ChemicalVAE latent space to scVI latent space
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
	cINN module connecting ChemicalVAE latent space and cell state
	covariates to scVI latent space
	"""
	def __init__(self,
				 configured_flow,
				 first_stage_data,
				 cond_stage_data,
				 model_con,
				 scvi_model,
				 std_model,
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
	cINN module connecting ChemicalVAE latent space to VAE latent space
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

# 			print(
# 				"[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f]"
# 				% (epoch, n_epochs, batch_idx + 1, len(test_loader), train_loss, test_loss)
# 			)

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



