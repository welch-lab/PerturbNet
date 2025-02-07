#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import sys
import torch.nn.functional as F
import torch.nn as nn
from perturbnet.pytorch_scvi.distributions import *

class ConcatDataset(torch.utils.data.Dataset):
	"""
	data structure with sample indices of two datasets
	"""
	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, i):
		return tuple(d[i] for d in self.datasets)

	def __len__(self):
		return min(len(d) for d in self.datasets)

class scvi_predictive_z:
	"""
	class to generate the gene expression data from latent variables of scVI
	"""
	def __init__(self, model):
		super().__init__()
		self.model = model

	def one_hot(self, index, n_cat):
		onehot = torch.zeros(index.size(0), n_cat, device = index.device)
		onehot.scatter_(1, index.type(torch.long), 1)
		return onehot.type(torch.float32)


	def decoder_inference(self, z, library, batch_index = None, y = None, n_samples = 1, 
		transform_batch = None):
		"""
		a function employed on the scVI.model object, currently only allow n_samples == 1
		"""
		if transform_batch is not None: 
			dec_batch_index = transform_batch * torch.ones_like(batch_index)
		else:
			dec_batch_index = batch_index

		px_scale, px_r, px_rate, px_dropout = self.model.model.decoder(self.model.model.dispersion, z, library, dec_batch_index, y)
		if self.model.model.dispersion == "gene-label":
			px_r = F.linear(
				self.one_hot(y, self.model.model.n_labels), self.model.model.px_r
				) # px_r gets transposed - last dimension is nb genes
		elif self.model.model.dispersion == "gene-batch":
			px_r = F.linear(self.one_hot(dec_batch_index, self.model.model.n_batch), self.model.model.px_r)
		elif self.model.model.dispersion == "gene":
			px_r = self.model.model.px_r
		px_r = torch.exp(px_r)

		return dict(
			px_scale = px_scale, 
			px_r = px_r, 
			px_rate = px_rate, 
			px_dropout = px_dropout)


	@torch.no_grad()
	def posterior_predictive_sample_from_Z(
		self,
		z_sample,
		l_sample, 
		n_samples: int = 1, 
		batch_size = None
		):

		if self.model.model.gene_likelihood not in ["zinb", "nb", "poisson"]:
			raise ValueError("Invalid gene_likelihood.")

		if batch_size is None:
			batch_size = 32
		
		data_loader = torch.utils.data.DataLoader(
				ConcatDataset(z_sample, l_sample),
				batch_size = batch_size, 
				shuffle = False)

		x_new = []
		for batch_idx, (batch_z, batch_l) in enumerate(data_loader):

			labels = None # currently only support unsupervised learning
			
			outputs = self.decoder_inference(
				batch_z, batch_l, batch_index = batch_idx, y = labels, n_samples = n_samples
			)
			px_r = outputs["px_r"]
			px_rate = outputs["px_rate"]
			px_dropout = outputs["px_dropout"]

			if self.model.model.gene_likelihood == "poisson":
				l_train = px_rate
				l_train = torch.clamp(l_train, max=1e8)
				dist = torch.distributions.Poisson(
					l_train
				)  # Shape : (n_samples, n_cells_batch, n_genes)
			elif self.model.model.gene_likelihood == "nb":
				dist = NegativeBinomial(mu = px_rate, theta = px_r)
			elif self.model.model.gene_likelihood == "zinb":
				dist = ZeroInflatedNegativeBinomial(
					mu = px_rate, theta = px_r, zi_logits = px_dropout
				)

			else:
				raise ValueError(
					"{} reconstruction error not handled right now".format(
						self.model.model.gene_likelihood
					)
				)

			if n_samples > 1:
				exprs = dist.sample().permute(
					[1, 2, 0]
				)  # Shape : (n_cells_batch, n_genes, n_samples)
			else:
				exprs = dist.sample()

			x_new.append(exprs.cpu())
		x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)

		return x_new.numpy()

	@torch.no_grad()
	def posterior_predictive_sample_from_Z_with_y(
		self,
		z_sample,
		l_sample,
		y_sample,
		n_samples: int = 1,
		batch_size = None
		):

		if self.model.model.gene_likelihood not in ["zinb", "nb", "poisson"]:
			raise ValueError("Invalid gene_likelihood.")

		if batch_size is None:
			batch_size = 32

		data_loader = torch.utils.data.DataLoader(
				ConcatDataset(z_sample, l_sample, y_sample),
				batch_size = batch_size,
				shuffle = False)

		x_new = []
		for batch_idx, (batch_z, batch_l, batch_y) in enumerate(data_loader):

			outputs = self.decoder_inference(
				batch_z, batch_l, batch_index = batch_idx, y = batch_y, n_samples = n_samples
			)
			px_r = outputs["px_r"]
			px_rate = outputs["px_rate"]
			px_dropout = outputs["px_dropout"]

			if self.model.model.gene_likelihood == "poisson":
				l_train = px_rate
				l_train = torch.clamp(l_train, max=1e8)
				dist = torch.distributions.Poisson(
					l_train
				)  # Shape : (n_samples, n_cells_batch, n_genes)
			elif self.model.model.gene_likelihood == "nb":
				dist = NegativeBinomial(mu = px_rate, theta = px_r)
			elif self.model.model.gene_likelihood == "zinb":
				dist = ZeroInflatedNegativeBinomial(
					mu = px_rate, theta = px_r, zi_logits = px_dropout
				)

			else:
				raise ValueError(
					"{} reconstruction error not handled right now".format(
						self.model.model.gene_likelihood
					)
				)

			if n_samples > 1:
				exprs = dist.sample().permute(
					[1, 2, 0]
				)  # Shape : (n_cells_batch, n_genes, n_samples)
			else:
				exprs = dist.sample()

			x_new.append(exprs.cpu())
		x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)

		return x_new.numpy()

	@torch.no_grad()
	def posterior_predictive_sample_from_Z_with_batch(
		self,
		z_sample,
		l_sample,
		batch_sample,
		n_samples: int = 1,
		batch_size = None
		):

		if self.model.model.gene_likelihood not in ["zinb", "nb", "poisson"]:
			raise ValueError("Invalid gene_likelihood.")

		if batch_size is None:
			batch_size = 32

		data_loader = torch.utils.data.DataLoader(
				ConcatDataset(z_sample, l_sample, batch_sample),
				batch_size = batch_size,
				shuffle = False)

		x_new = []
		for batch_idx, (batch_z, batch_l, batch_batch) in enumerate(data_loader):

			labels = None # currently only support unsupervised learning

			outputs = self.decoder_inference(
				batch_z, batch_l, batch_index = batch_batch.view(batch_batch.shape[0], -1), y = labels, n_samples = n_samples
			)
			px_r = outputs["px_r"]
			px_rate = outputs["px_rate"]
			px_dropout = outputs["px_dropout"]

			if self.model.model.gene_likelihood == "poisson":
				l_train = px_rate
				l_train = torch.clamp(l_train, max=1e8)
				dist = torch.distributions.Poisson(
					l_train
				)  # Shape : (n_samples, n_cells_batch, n_genes)
			elif self.model.model.gene_likelihood == "nb":
				dist = NegativeBinomial(mu = px_rate, theta = px_r)
			elif self.model.model.gene_likelihood == "zinb":
				dist = ZeroInflatedNegativeBinomial(
					mu = px_rate, theta = px_r, zi_logits = px_dropout
				)

			else:
				raise ValueError(
					"{} reconstruction error not handled right now".format(
						self.model.model.gene_likelihood
					)
				)

			if n_samples > 1:
				exprs = dist.sample().permute(
					[1, 2, 0]
				)  # Shape : (n_cells_batch, n_genes, n_samples)
			else:
				exprs = dist.sample()

			x_new.append(exprs.cpu())
		x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)

		return x_new.numpy()
