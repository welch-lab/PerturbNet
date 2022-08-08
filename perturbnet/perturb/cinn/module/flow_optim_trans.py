#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from plotnine import *
from scipy import linalg
import random


def tranUmap(latent_base, latent_new):
	""" function to umap embed new latents """
	trans = umap.UMAP(random_state = 42, min_dist = 0.5, n_neighbors = 30).fit(latent_base)
	x_embedded = trans.transform(np.concatenate((latent_base, latent_new)))

	return x_embedded


def plotUmap(x_embedded, label_base, label_new, n_label_base, ifPlotBoth = True):
	""" Plot the Umap of latents """

	category = np.append([label_base] * n_label_base,
						 [label_new] * (x_embedded.shape[0] - n_label_base))
	df_umap_pr = x_embedded
	df_umap_pr = pd.DataFrame(df_umap_pr)
	df_umap_pr['x-umap'] = x_embedded[:, 0]
	df_umap_pr['y-umap'] = x_embedded[:, 1]
	df_umap_pr['Category'] = category

	min_x, min_y = np.floor(df_umap_pr['x-umap'].min()), np.floor(df_umap_pr['y-umap'].min())
	max_x, max_y = np.ceil(df_umap_pr['x-umap'].max()), np.ceil(df_umap_pr['y-umap'].max())

	if ifPlotBoth:
		plotData = df_umap_pr
	else:
		plotData = df_umap_pr.iloc[n_label_base:, :]

	chart_pr = ggplot(plotData, aes(x='x-umap', y='y-umap', colour='Category')) \
			   + geom_point(size=0.1, alpha=1) + labs(x="UMAP1", y="UMAP2") \
			   + xlim(min_x, max_x) + ylim(min_y, max_y) + theme_bw() \
			   + theme(panel_background=element_rect(fill='white'),
					   title=element_text(size=25),
					   axis_title_x=element_text(size=15),
					   axis_title_y=element_text(size=15),
					   legend_title=element_text(size=20),
					   legend_text=element_text(size=20),
					   axis_text_y=element_blank(),
					   axis_text_x=element_blank(),
					   axis_ticks_major_y=element_blank(),
					   axis_ticks_major_x=element_blank(),
					   panel_grid=element_blank()) \
			   + guides(colour=guide_legend(nrow=3, override_aes={"size": 10}))
	return chart_pr


def plotUmapOneCat(x_embedded, label_new, ifPlotBoth = True):
	"""Plot the Umap of latents for one category"""

	category = [label_new] * (x_embedded.shape[0])
	df_umap_pr = x_embedded
	df_umap_pr = pd.DataFrame(df_umap_pr)
	df_umap_pr['x-umap'] = x_embedded[:, 0]
	df_umap_pr['y-umap'] = x_embedded[:, 1]
	df_umap_pr['Category'] = category

	min_x, min_y = np.floor(df_umap_pr['x-umap'].min()), np.floor(df_umap_pr['y-umap'].min())
	max_x, max_y = np.ceil(df_umap_pr['x-umap'].max()), np.ceil(df_umap_pr['y-umap'].max())

	chart_pr = ggplot(df_umap_pr, aes(x='x-umap', y='y-umap', colour='Category')) \
			   + geom_point(size=0.1, alpha=1) + labs(x="UMAP1", y="UMAP2") \
			   + xlim(min_x, max_x) + ylim(min_y, max_y) + theme_bw() \
			   + theme(panel_background=element_rect(fill='white'),
					   title=element_text(size=25),
					   axis_title_x=element_text(size=15),
					   axis_title_y=element_text(size=15),
					   legend_title=element_text(size=20),
					   legend_text=element_text(size=20),
					   axis_text_y=element_blank(),
					   axis_text_x=element_blank(),
					   axis_ticks_major_y=element_blank(),
					   axis_ticks_major_x=element_blank(),
					   panel_grid=element_blank()) \
			   + guides(colour=guide_legend(nrow=3, override_aes={"size": 10}))
	return chart_pr


class Wdistance:
	"""
	Class of calculating squared Wasserstein-2/FID distance, 
	with an option ifClosedForm to use the closed-form expression
	"""

	def __init__(self):
		super().__init__()

	def calculate_statistics(self, numpy_data):
		mu = np.mean(numpy_data, axis=0)
		sigma = np.cov(numpy_data, rowvar=False)
		return mu, sigma

	def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, ifClosedForm = True, eps=1e-6):
		diff = mu1 - mu2
		if ifClosedForm:
			s1Root, _ = linalg.sqrtm(sigma1, disp=False)
			s1rS2S1r = s1Root.dot(sigma2.dot(s1Root))
			s1rS2S1r_root, _ = linalg.sqrtm(s1rS2S1r, disp=False)
			covmean = s1Root.dot(s1rS2S1r_root.dot(np.linalg.inv(s1Root)))
		else:
			covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
		if not np.isfinite(covmean).all():
			msg = (
					'fid calculation produces singular product; '
					'adding %s to diagonal of cov estimates' % eps
			)
			print(msg)
			offset = np.eye(sigma1.shape[0]) * eps
			covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

		if np.iscomplexobj(covmean):
			if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
				m = np.max(np.abs(covmean.imag))
			# raise ValueError('Cell component {}'.format(m))
			covmean = covmean.real

			image_error = 1
		else:
			image_error = 0

		tr_covmean = np.trace(covmean)

		return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean), image_error

	def calculate_fid_score(self, real_data, fake_data):

		if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
			return -1, 1

		data1, data2 = real_data, fake_data

		m1, s1 = self.calculate_statistics(data1)
		m2, s2 = self.calculate_statistics(data2)
		fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

		return fid_value, image_error

	def calculate_r_square(self, real_data, fake_data):
		x = np.average(fake_data, axis=0)
		y = np.average(real_data, axis=0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2

	def calculate_r_square_var(self, real_data, fake_data):
		x = np.var(fake_data, axis=0)
		y = np.var(real_data, axis=0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2

class Wasserstein2Gaussian:
	"""
	Pytorch class of calculating squared Wasserstein-2/FID distance
	"""

	def __init__(self):
		super().__init__()

	def calculate_statistics(self, torch_data):
		mu = torch.mean(torch_data, 0)
		sigma = torch.cov(torch_data.T)
		return mu, sigma

	def symsqrt(self, a, cond=None, return_rank=False):
		"""Computes the symmetric square root of a positive definite matrix
		Thanks to https://github.com/pytorch/pytorch/issues/25481
		"""
		s, u = torch.symeig(a, eigenvectors=True)
		cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

		if cond in [None, -1]:
			cond = cond_dict[a.dtype]

		above_cutoff = (abs(s) > cond * torch.max(abs(s)))

		psigma_diag = torch.sqrt(s[above_cutoff])
		u = u[:, above_cutoff]

		B = u @ torch.diag(psigma_diag) @ u.t()
		if return_rank:
			return B, len(psigma_diag)
		else:
			return B

	def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
		diff = torch.subtract(mu1, mu2)
		covmean = self.symsqrt(torch.matmul(sigma1, sigma2))
		if not torch.isfinite(covmean).numpy().all():
			msg = (
					'fid calculation produces singular product; '
					'adding %s to diagonal of cov estimates' % eps
			)
			print(msg)
			offset = torch.eye(sigma1.shape[0]) * eps
			covmean = self.symsqrt(torch.matmul(sigma1 + offset, sigma2 + offset))

		if np.iscomplexobj(covmean.detach().numpy()):
			covmean = covmean.real
			image_error = 1
		else:
			image_error = 0

		tr_covmean = torch.trace(covmean)

		return (torch.dot(diff, diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean), image_error

	def calculate_fid_score(self, real_data, fake_data, pca_data_fit=None, if_dataPC=False):

		if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
			return -1, 1

		data1, data2 = fake_data, real_data

		m1, s1 = self.calculate_statistics(data1)
		m2, s2 = self.calculate_statistics(data2)
		fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

		return fid_value, image_error



class Wasserstein2GaussianGPU:
	"""
	Pytorch GPU class of calculating squared Wasserstein-2/FID distance within python version 3.8
	"""

	def __init__(self):
		super().__init__()

	def calculate_statistics(self, torch_data):
		mu = torch.mean(torch_data, 0)
		sigma = torch.cov(torch_data.T)
		return mu, sigma

	def symsqrt(self, a, cond=None, return_rank=False):
		"""Computes the symmetric square root of a positive definite matrix
		Thanks to https://github.com/pytorch/pytorch/issues/25481
		"""
		s, u = torch.symeig(a, eigenvectors=True)
		cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

		if cond in [None, -1]:
			cond = cond_dict[a.dtype]

		above_cutoff = (abs(s) > cond * torch.max(abs(s)))

		psigma_diag = torch.sqrt(s[above_cutoff])
		u = u[:, above_cutoff]

		B = u @ torch.diag(psigma_diag) @ u.t()
		if return_rank:
			return B, len(psigma_diag)
		else:
			return B

	def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
		diff = torch.subtract(mu1, mu2)
		covmean = self.symsqrt(torch.matmul(sigma1, sigma2))
		if not torch.isfinite(covmean).cpu().numpy().all():
			msg = (
					'fid calculation produces singular product; '
					'adding %s to diagonal of cov estimates' % eps
			)
			print(msg)
			offset = torch.eye(sigma1.shape[0]) * eps
			covmean = self.symsqrt(torch.matmul(sigma1 + offset, sigma2 + offset))

		if np.iscomplexobj(covmean.detach().cpu().numpy()):
			covmean = covmean.real
			image_error = 1
		else:
			image_error = 0

		tr_covmean = torch.trace(covmean)

		return (torch.dot(diff, diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean), image_error

	def calculate_fid_score(self, real_data, fake_data, pca_data_fit=None, if_dataPC=False):

		if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
			return -1, 1

		data1, data2 = fake_data, real_data

		m1, s1 = self.calculate_statistics(data1)
		m2, s2 = self.calculate_statistics(data2)
		fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

		return fid_value, image_error


def torch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
	"""Estimates covariance matrix like numpy.cov"""
	# ensure at least 2D
	if x.dim() == 1:
		x = x.view(-1, 1)

	# treat each column as a data point, each row as a variable
	if rowvar and x.shape[0] != 1:
		x = x.t()

	if ddof is None:
		if bias == 0:
			ddof = 1
		else:
			ddof = 0

	w = aweights
	if w is not None:
		if not torch.is_tensor(w):
			w = torch.tensor(w, dtype=torch.float)
		w_sum = torch.sum(w)
		avg = torch.sum(x * (w/w_sum)[:,None], 0)
	else:
		avg = torch.mean(x, 0)

	# Determine the normalization
	if w is None:
		fact = x.shape[0] - ddof
	elif ddof == 0:
		fact = w_sum
	elif aweights is None:
		fact = w_sum - ddof
	else:
		fact = w_sum - ddof * torch.sum(w * w) / w_sum

	xm = x.sub(avg.expand_as(x))

	if w is None:
		X_T = xm.t()
	else:
		X_T = torch.mm(torch.diag(w), xm).t()

	c = torch.mm(X_T, xm)
	c = c / fact

	return c.squeeze()



class Wasserstein2GaussianGPUPy36:
	"""
	Pytorch GPU class of calculating squared Wasserstein-2/FID distance within python version 3.6
	"""
	def __init__(self):
		super().__init__()

	def calculate_statistics(self, torch_data):
		mu = torch.mean(torch_data, 0)
		sigma = torch_cov(torch_data)
		return mu, sigma

	def symsqrt(self, a, cond=None, return_rank=False):
		"""Computes the symmetric square root of a positive definite matrix
		Thanks to https://github.com/pytorch/pytorch/issues/25481
		"""
		s, u = torch.symeig(a, eigenvectors=True)
		cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

		if cond in [None, -1]:
			cond = cond_dict[a.dtype]

		above_cutoff = (abs(s) > cond * torch.max(abs(s)))

		psigma_diag = torch.sqrt(s[above_cutoff])
		u = u[:, above_cutoff]

		B = u @ torch.diag(psigma_diag) @ u.t()
		if return_rank:
			return B, len(psigma_diag)
		else:
			return B

	def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, ifClosedForm = True, eps=1e-6):
		diff = torch.subtract(mu1, mu2)
		if ifClosedForm:
			s1Root = self.symsqrt(sigma1)
			s1rS2S1r = torch.matmul(s1Root, torch.matmul(sigma2, s1Root))
			s1rS2S1r_root = self.symsqrt(s1rS2S1r)
			covmean = torch.matmul(s1Root, torch.matmul(s1rS2S1r_root, torch.inverse(s1Root)))
		else:
			covmean = self.symsqrt(torch.matmul(sigma1, sigma2))
		if not torch.isfinite(covmean).cpu().numpy().all():
			msg = (
					'fid calculation produces singular product; '
					'adding %s to diagonal of cov estimates' % eps
			)
			print(msg)
			offset = torch.eye(sigma1.shape[0]) * eps
			covmean = self.symsqrt(torch.matmul(sigma1 + offset, sigma2 + offset))

		if np.iscomplexobj(covmean.detach().cpu().numpy()):
			covmean = covmean.real
			image_error = 1
		else:
			image_error = 0

		tr_covmean = torch.trace(covmean)

		return (torch.dot(diff, diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean), image_error

	def calculate_fid_score(self, real_data, fake_data, pca_data_fit=None, if_dataPC=False):

		if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
			return -1, 1

		data1, data2 = fake_data, real_data

		m1, s1 = self.calculate_statistics(data1)
		m2, s2 = self.calculate_statistics(data2)
		fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

		return fid_value, image_error


class SCVI_OptimTransPy36(nn.Module):
	"""
	Class to translate some cells (latent values, perturbations) to their counterfactual
	latent values using a perturbation based on python version 3.6
	"""

	def __init__(self, n_latent, model, device):
		super().__init__()
		condition_new = torch.distributions.normal.Normal(0.0, 1.0).sample((1, n_latent))
		self.condition_new = nn.Parameter(condition_new)
		self.model = model
		self.device = device

	def generate_v(self, x, c):
		zz, _ = self.model.flow(x, c)
		return zz

	def generate_zprime(self, zz, cprime):
		return self.model.flow.reverse(zz, cprime)

	def trans_zprime(self, latent, condition, condition_new):
		zz = self.generate_v(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
							 torch.tensor(condition).float().to(self.device))

		return self.generate_zprime(zz, condition_new.float().to(self.device)).squeeze(-1).squeeze(-1)

	def forward(self, latent_start, condition):
		condition_new = self.condition_new.repeat(latent_start.shape[0], 1)
		return self.trans_zprime(latent_start, condition, condition_new)



def train_cinnTrans_model_gpu_py36(model, latent_start, condition, latent_target, optimizer,  device, iteration = 1000):
	"""
	Training module of continuous optimal translation based on python version 3.6
	"""
	wscore_cal = Wasserstein2GaussianGPUPy36()
	losses = []
	target_tensor = torch.tensor(latent_target).to(device)
	for i in range(iteration):
		preds = model(latent_start, condition)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		print("i = " + str(i) + ", loss = " + str(loss) + "\n")
	return losses

def train_cinnTrans_model_gpu_py36_crossValidate(model, latent_start_train, condition_train,
												 latent_start_test, condition_test,
												 latent_target, optimizer,  device, iteration = 1000):
	"""
	Training module of continuous optimal translation based on python version 3.6 
	with cross-validation (train-test splitting)
	"""
	wscore_cal = Wasserstein2GaussianGPUPy36()
	train_losses, test_losses = [], []
	target_tensor = torch.tensor(latent_target).to(device)
	for i in range(iteration):
		#model.train()
		preds = model(latent_start_train, condition_train)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_losses.append(loss.item())

		#model.eval()
		preds = model(latent_start_test, condition_test)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		test_losses.append(loss.item())

	return train_losses, test_losses



class SCVI_OptimTrans(nn.Module):
	"""
	Class to translate some cells (latent values, perturbations) to their counterfactual
	latent values using a perturbation based on python version 3.8
	"""
	def __init__(self, n_latent, model, device):
		super().__init__()
		condition_new = torch.distributions.normal.Normal(0.0, 1.0).sample((1, n_latent))
		self.condition_new = nn.Parameter(condition_new)
		self.model = model
		self.device = device

	def generate_v(self, x, c):
		zz, _ = self.model.flow(x, c)
		return zz

	def generate_zprime(self, zz, cprime):
		return self.model.flow.reverse(zz, cprime)

	def trans_zprime(self, latent, condition, condition_new):
		zz = self.generate_v(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
							 torch.tensor(condition).float().to(self.device))

		return self.generate_zprime(zz, condition_new.float().to(self.device)).squeeze(-1).squeeze(-1)

	def forward(self, latent_start, condition):
		condition_new = torch.tile(self.condition_new, (latent_start.shape[0], 1))
		return self.trans_zprime(latent_start, condition, condition_new)


class SCVI_State_OptimTrans(nn.Module):
	"""
	Class to translate some cells (latent values, perturbations) to their counterfactual
	latent values using a perturbation and their cell state covariates based on python version 3.8
	"""

	def __init__(self, n_latent, model, device):
		super().__init__()
		condition_new = torch.distributions.normal.Normal(0.0, 1.0).sample((1, n_latent))
		self.condition_new = nn.Parameter(condition_new)
		self.model = model
		self.device = device

	def generate_v(self, x, c):
		zz, _ = self.model.flow(x, c)
		return zz

	def generate_zprime(self, zz, cprime):
		return self.model.flow.reverse(zz, cprime)

	def trans_zprime(self, latent, condition, condition_new):
		zz = self.generate_v(torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
							 torch.tensor(condition).float().to(self.device))

		return self.generate_zprime(zz, condition_new.float().to(self.device)).squeeze(-1).squeeze(-1)

	def forward(self, latent_start, condition, state):
		condition_new_part = torch.tile(self.condition_new, (latent_start.shape[0], 1)).float().to(self.device)
		condition_new = torch.cat((condition_new_part, torch.tensor(state).float().to(self.device)), 1)
		return self.trans_zprime(latent_start, condition, condition_new)


def train_cinnTrans_model(model, latent_start, condition, latent_target, optimizer, iteration = 1000):
	"""
	Training module of continuous optimal translation based on python version 3.8
	"""

	wscore_cal = Wasserstein2Gaussian()
	losses = []
	target_tensor = torch.tensor(latent_target)
	for i in range(iteration):
		preds = model(latent_start, condition)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		print("i = " + str(i) + ", loss = " + str(loss) + "\n")
	return losses

def train_cinnTrans_model_gpu(model, latent_start, condition, latent_target, optimizer,  device, iteration = 1000):
	"""
	Training module of continuous optimal translation based on python version 3.8 and GPU
	"""
	wscore_cal = Wasserstein2GaussianGPU()
	losses = []
	target_tensor = torch.tensor(latent_target).to(device)
	for i in range(iteration):
		preds = model(latent_start, condition)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		print("i = " + str(i) + ", loss = " + str(loss) + "\n")
	return losses


def train_cinnTrans_model_gpu_crossValidate(model, latent_start_train,
											condition_train, latent_start_test,
											condition_test,
											latent_target,
											optimizer,  device, iteration = 1000):
	"""
	Training module of continuous optimal translation based on python version 3.8 and GPU
	with cross-validation (train-test splitting)
	"""
	wscore_cal = Wasserstein2GaussianGPU()
	train_losses, test_losses = [], []
	target_tensor = torch.tensor(latent_target).to(device)
	for i in range(iteration):
		#model.train()
		preds = model(latent_start_train, condition_train)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())

		#model.eval()
		preds = model(latent_start_test, condition_test)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		test_losses.append(loss.item())
	return train_losses, test_losses

def train_cinnTrans_state_model_gpu(model, latent_start, condition, latent_target, state_start, optimizer,  device, iteration = 1000):
	"""
	Training module of continuous optimal translation with cell state covariates
	based on python version 3.8 and GPU
	with cross-validation (train-test splitting)
	"""
	wscore_cal = Wasserstein2GaussianGPU()
	losses = []
	target_tensor = torch.tensor(latent_target).to(device)
	for i in range(iteration):
		preds = model(latent_start, condition, state_start)
		loss, _ = wscore_cal.calculate_fid_score(target_tensor, preds)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		print("i = " + str(i) + ", loss = " + str(loss) + "\n")
	return losses


def train_cinnTrans_model_gpu_sgd(model, latent_start, condition, latent_target, optimizer, device, batch_size = 128,
								  n_epochs = 10):
	"""
	Training module of continuous optimal translation with cell state covariates
	based on python version 3.8 and GPU
	with minibatch optimization
	"""
	assert latent_start.shape[0] == condition.shape[0]

	wscore_cal = Wasserstein2GaussianGPU()
	losses = []
	target_tensor = torch.tensor(latent_target).to(device)
	n_larger = max(latent_start.shape[0], latent_target.shape[0])
	for i in range(n_epochs):
		loss_epoch = 0.0
		for j in range(n_larger // batch_size):
			indices_start = random.sample(range(latent_start.shape[0]), batch_size)
			indices_target = random.sample(range(latent_target.shape[0]), batch_size)

			preds = model(latent_start[indices_start], condition[indices_start])
			loss, _ = wscore_cal.calculate_fid_score(target_tensor[indices_target], preds)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_epoch += loss.item()
		losses.append(loss_epoch)
		print("i = " + str(i) + ", loss = " + str(loss) + "\n")
	return losses