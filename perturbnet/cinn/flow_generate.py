#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SCVIZ_CheckNet2Net:
	"""Class to use PerturbNet to predict cellular representations and 
	count responses (scVI) from perturbation representations
	"""
	def __init__(self, model, device, scvi_model_decode):
		super().__init__()
		self.model = model
		self.device = device
		self.scvi_model_decode = scvi_model_decode

	@torch.no_grad()
	def sample_data(self, condition_data, library_data, batch_size = 50):
		#torch.manual_seed(2021)        
		sampled_z = self.model.sample_conditional(torch.tensor(condition_data)\
			.float().to(self.device)).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		sampled_data = self.scvi_model_decode.posterior_predictive_sample_from_Z(sampled_z, library_data, batch_size = batch_size)

		return sampled_z, sampled_data

	@torch.no_grad()
	def recon_v(self, latent, condition):
		recon_val, _ = self.model.flow(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
						torch.tensor(condition).float().to(self.device))
		return recon_val.squeeze(-1).squeeze(-1).cpu().detach().numpy()

	@torch.no_grad()
	def trans_data(self, latent, condition, condition_new, library_data, batch_size = 50):
		trans_z = self.model.generate_zprime(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1), 
											 torch.tensor(condition).float().to(self.device),
											 torch.tensor(condition_new).float().to(self.device)
			).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		trans_data = self.scvi_model_decode.posterior_predictive_sample_from_Z(trans_z, library_data, batch_size = batch_size)

		return trans_z, trans_data

	@torch.no_grad()
	def recon_data(self, latent, condition, library_data, batch_size = 50):

		rec_z = self.model.generate_zrec(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1), 
								 torch.tensor(condition).float().to(self.device))\
			.squeeze(-1).squeeze(-1).cpu().detach().numpy()

		rec_data = self.scvi_model_decode.posterior_predictive_sample_from_Z(rec_z, library_data, batch_size = batch_size)

		return rec_z, rec_data

	@torch.no_grad()
	def recon_data_with_y(self, latent, condition, library_data, y_data, batch_size = 50):

		rec_z = self.model.generate_zrec(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
								 torch.tensor(condition).float().to(self.device))\
			.squeeze(-1).squeeze(-1).cpu().detach().numpy()

		rec_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_y(rec_z, library_data, y_data, batch_size = batch_size)

		return rec_z, rec_data

	@torch.no_grad()
	def sample_data_with_y(self, condition_data, library_data, y_data, batch_size = 50):
		sampled_z = self.model.sample_conditional(torch.tensor(condition_data)\
			.float().to(self.device)).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		sampled_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_y(sampled_z, library_data, y_data, batch_size = batch_size)

		return sampled_z, sampled_data

	@torch.no_grad()
	def trans_data_with_y(self, latent, condition, condition_new, library_data, y_data, batch_size = 50):
		trans_z = self.model.generate_zprime(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
											 torch.tensor(condition).float().to(self.device),
											 torch.tensor(condition_new).float().to(self.device)
			).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		trans_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_y(trans_z, library_data, y_data, batch_size = batch_size)

		return trans_z, trans_data


	@torch.no_grad()
	def recon_data_with_batch(self, latent, condition, library_data, batch_data, batch_size = 50):

		rec_z = self.model.generate_zrec(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
								 torch.tensor(condition).float().to(self.device))\
			.squeeze(-1).squeeze(-1).cpu().detach().numpy()

		rec_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_batch(rec_z, library_data, batch_data, batch_size = batch_size)

		return rec_z, rec_data

	@torch.no_grad()
	def sample_data_with_batch(self, condition_data, library_data, batch_data, batch_size = 50):
		sampled_z = self.model.sample_conditional(torch.tensor(condition_data)\
			.float().to(self.device)).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		sampled_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_batch(sampled_z, library_data, batch_data, batch_size = batch_size)

		return sampled_z, sampled_data

	@torch.no_grad()
	def trans_data_with_batch(self, latent, condition, condition_new, library_data, batch_data, batch_size = 50):
		trans_z = self.model.generate_zprime(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
											 torch.tensor(condition).float().to(self.device),
											 torch.tensor(condition_new).float().to(self.device)
			).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		trans_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_batch(trans_z, library_data, batch_data, batch_size = batch_size)

		return trans_z, trans_data



class TFVAEZ_CheckNet2Net:
	"""Class to use PerturbNet to predict cellular representations and 
	normalized responses (VAE) from perturbation representations
	"""
	def __init__(self, model, device, tf_sess, tf_x_rec_data, tf_z_latent, is_training):
		super().__init__()
		self.model = model
		self.device = device
		self.tf_sess = tf_sess 
		self.tf_x_rec_data = tf_x_rec_data
		self.tf_z_latent = tf_z_latent
		self.is_training = is_training

	@torch.no_grad()
	def sample_data(self, condition_data):

		sampled_z = self.model.sample_conditional(torch.tensor(condition_data)\
			.float().to(self.device)).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		feed_dict = {self.tf_z_latent: sampled_z, self.is_training: False}
		sampled_data = self.tf_sess.run(self.tf_x_rec_data, feed_dict = feed_dict)

		return sampled_z, sampled_data

	@torch.no_grad()
	def recon_v(self, latent, condition):
		recon_val, _ = self.model.flow(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
						torch.tensor(condition).float().to(self.device))
		return recon_val.squeeze(-1).squeeze(-1).cpu().detach().numpy()

	@torch.no_grad()
	def trans_data(self, latent, condition, condition_new):
		trans_z = self.model.generate_zprime(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1), 
											 torch.tensor(condition).float().to(self.device),
											 torch.tensor(condition_new).float().to(self.device)
			).squeeze(-1).squeeze(-1).cpu().detach().numpy()

		feed_dict = {self.tf_z_latent: trans_z, self.is_training: False}
		trans_data = self.tf_sess.run(self.tf_x_rec_data, feed_dict = feed_dict)

		return trans_z, trans_data
	
	@torch.no_grad()
	def recon_data(self, latent, condition):

		rec_z = self.model.generate_zrec(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1), 
								 torch.tensor(condition).float().to(self.device))\
			.squeeze(-1).squeeze(-1).cpu().detach().numpy()

		feed_dict = {self.tf_z_latent: rec_z, self.is_training: False}
		rec_data = self.tf_sess.run(self.tf_x_rec_data, feed_dict = feed_dict)

		return rec_z, rec_data


