#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../..')
import os
import time
import random

import torch
import torch.nn.functional as F
import torch.nn as nn

from captum.attr import IntegratedGradients
from pytorch_scvi.distributions import *
from pytorch_scvi.scvi_generate_z import *

import tensorflow as tf
from tensorflow import distributions as ds

from perturbnet.perturb.util import *
from perturbnet.perturb.cinn.modules.flow import *
from perturbnet.perturb.chemicalvae.chemicalVAE import *
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from perturbnet.perturb.knn.baseline_knn_lincs import *
from perturbnet.perturb.cinn.modules.flow_generate import TFVAEZ_CheckNet2Net
from perturbnet.perturb.cinn.modules.flow_optim_trans import *

# UMAP of the data
import umap
from plotnine import *
import matplotlib
import matplotlib.pyplot as plt


class BinaryCellStatesClass(nn.Module):
	""" Class to predict the label for two latent spaces """
	def __init__(self, z_dim = 10, hidden_dim = 32, prob_drop = 0.1):
		super().__init__()
		self.bn1 = nn.BatchNorm1d(hidden_dim)
		self.linear_1 = nn.Linear(z_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, 1)
		self.relu = nn.ReLU()
		self.dropout1 = nn.Dropout(p = prob_drop)

	def forward(self, x):
		x = self.relu(self.linear_1(x))
		x = self.bn1(x)
		x = self.dropout1(x)
		x = torch.sigmoid(self.linear_2(x))
		return x

class FlowResizeLabelClass(nn.Module):
	"""Class to generate cellular representations via PerturbNet from perturbation onehot encodings"""
	def __init__(self,
				 model,
				 model_g,
				 model_class,
				 std_model,
				 zDim,
				 yDim,
				 n_seq,
				 n_vol,
				 device):
		super().__init__()
		self.model = model
		self.model_g = model_g
		self.model_class = model_class
		self.std_model = std_model
		self.zDim = zDim
		self.yDim = yDim
		self.n_seq = n_seq
		self.n_vol = n_vol
		self.device = device

	def generte_zprime(self,  x, c, cprime):
		zz, _ = self.model.flow(x, c)
		zprime = self.model.flow.reverse(zz, cprime)
		return zprime

	def forward(self, input_data, batch_size = 50):
		latent = input_data[:, :self.zDim]

		condition = input_data[:, self.zDim:(self.zDim + self.yDim)]
		trt_onehot = input_data[:, (self.zDim + self.yDim):]
		trt_onehot = trt_onehot.view(trt_onehot.size(0), self.n_seq, self.n_vol)

		_, _, _, embdata_torch_sub = self.model_g(trt_onehot.float())
		condition_new = self.std_model.standardize_z_torch(embdata_torch_sub)

		trans_z = self.generte_zprime(latent.float().to(self.device).unsqueeze(-1).unsqueeze(-1),
											 condition.float().to(self.device),
											 condition_new.float().to(self.device)
			).squeeze(-1).squeeze(-1)#.cpu().detach().numpy()

		trans_z_class = self.model_class(trans_z)

		return trans_z_class


class FlowResizeYLabelClass(nn.Module):
	"""Class to generate cellular representations via PerturbNet from perturbation representations"""
	def __init__(self,
				 model,
				 model_g,
				 model_class,
				 std_model,
				 zDim,
				 yDim,
				 n_seq,
				 n_vol,
				 device):
		super().__init__()
		self.model = model
		self.model_g = model_g
		self.model_class = model_class
		self.std_model = std_model
		self.zDim = zDim
		self.yDim = yDim
		self.n_seq = n_seq
		self.n_vol = n_vol
		self.device = device

	def generte_zprime(self,  x, c, cprime):
		zz, _ = self.model.flow(x, c)
		zprime = self.model.flow.reverse(zz, cprime)
		return zprime

	def forward(self, input_data, batch_size = 50):
		latent = input_data[:, :self.zDim]

		condition = input_data[:, self.zDim:(self.zDim + self.yDim)]
		trt_onehot = input_data[:, (self.zDim + self.yDim):]

		#trt_onehot = trt_onehot.view(trt_onehot.size(0), self.n_seq, self.n_vol)
		#_, _, _, embdata_torch_sub = self.model_g(trt_onehot.float())
		#condition_new = self.std_model.standardize_z_torch(embdata_torch_sub)

		condition_new = trt_onehot.float()

		trans_z = self.generte_zprime(latent.float().to(self.device).unsqueeze(-1).unsqueeze(-1),
											 condition.float().to(self.device),
											 condition_new.float().to(self.device)
			).squeeze(-1).squeeze(-1)#.cpu().detach().numpy()

		trans_z_class = self.model_class(trans_z)

		return trans_z_class


def ig_b_score_compute(ig, input_data, baseline_null, target, batch_size = 32, ifPlot = False, plot_save_file = '.'):
	"""
	Integrated gradients attributions of onehot encodings of perturbations
	"""
	n = input_data.shape[0]
	n_batches = n // batch_size
	if n_batches * batch_size < n:
		n_batches += 1

	attr_ig = None
	for i in range(n_batches):
		start = i * batch_size
		end = min(start + batch_size, n)

		attributions, delta_ig = ig.attribute(input_data[start:end],
											  baseline_null[start:end],
											  target = target,
											  return_convergence_delta = True)
		attributions = attributions[:, (10 + 196):].view(attributions.size(0), 120, 35)
		if attr_ig is None:
			attr_ig = attributions.cpu().detach().numpy()
		else:
			attr_ig = np.concatenate((attr_ig, attributions.cpu().detach().numpy()), axis = 0)

	if ifPlot:
		newfig = plt.figure(figsize = (20, 10))
		plt.imshow(attr_ig.mean(0).T, cmap = 'viridis')
		plt.colorbar()
		newfig.savefig(plot_save_file)

	return attr_ig.mean(0)


def ig_y_score_compute(ig, input_data, baseline_null, target, batch_size = 32, ifPlot = False, plot_save_file = '.'):
	"""
	Integrated gradients attributions of representations of perturbations
	"""
	n = input_data.shape[0]
	n_batches = n // batch_size
	if n_batches * batch_size < n:
		n_batches += 1

	attr_ig = None
	for i in range(n_batches):
		start = i * batch_size
		end = min(start + batch_size, n)

		attributions, delta_ig = ig.attribute(input_data[start:end],
											  baseline_null[start:end],
											  target = target,
											  return_convergence_delta = True)
		attributions = attributions[:, (10 + 196):]#.view(attributions.size(0), 120, 35)
		if attr_ig is None:
			attr_ig = attributions.cpu().detach().numpy()
		else:
			attr_ig = np.concatenate((attr_ig, attributions.cpu().detach().numpy()), axis = 0)

# 	if ifPlot:
# 		newfig = plt.figure(figsize = (20, 10))
# 		plt.imshow(attr_ig.mean(0).T, cmap = 'viridis')
# 		plt.colorbar()
# 		newfig.savefig(plot_save_file)

	return attr_ig.mean(0)

if __name__ == "__main__":

	torch.manual_seed(123)
	np.random.seed(123)
	path_save = "output"
	n_per = 3000
	top_n = 12 # top n perturbations for the clustered drug scores
	compared_scenarios = [""]
	path_binaryClass_model_list = "model/"

	path_cinn_model = ""
	path_start_per = ""
	path_data = ""
	path_chemvae_model = ""

	path_vae_model_eval = ""
	path_vae_model_train = ""

	path_lincs_onehot = ""
	path_chem_onehot = ""
	path_std_param = ""

	usedata = np.load(os.path.join(path_data, "data.npy"))

	## trts
	trt_list = np.load(os.path.join(path_lincs_onehot,
									"Smiles.npy"))

	## onehot
	data_lincs_onehot = np.load(os.path.join(path_lincs_onehot,
											 "SmilesOneHot.npy"))
	data_chem_onehot = np.load(path_chem_onehot)
	## meta information
	idx_to_train = np.load(os.path.join(path_lincs_onehot,
										"idx.npy"))
	input_ltpm_label = pd.read_csv(
		os.path.join(path_data, "PerturbMeta.csv"))

	perturb_with_onehot_overall = np.array(list(input_ltpm_label["canonical_smiles"]))
	input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]

	list_canonSmiles = list(input_ltpm_label["canonical_smiles"])
	# indicesWithOnehot = [i for i in range(len(list_canonSmiles)) if list_canonSmiles[i] in set(trt_list)]
	indicesWithOnehot = np.in1d(list_canonSmiles, trt_list)

	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train][indicesWithOnehot]

	removed_all_pers = np.load(os.path.join(path_lincs_onehot, "RemovedPerturbs.npy"))
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	input_ltpm_label1 = input_ltpm_label.iloc[kept_indices, :]
	input_ltpm_label1.index = list(range(input_ltpm_label1.shape[0]))

	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]
	list_string_interest = list(pd.Series(perturb_with_onehot_kept).value_counts().head(top_n).keys())
	scenario = compared_scenarios[0]
	trt_type_base, trt_type, fitted = scenario.split("_")
	list_string_interest += [trt_type_base, trt_type, fitted]

	vae_train = VAE(num_cells_train=usedata.shape[0], x_dimension=usedata.shape[1], learning_rate=1e-4,
					BNTrainingMode=False)
	vae_train.restore_model(path_vae_model_train)

	vae_eval = VAE(num_cells_train=usedata.shape[0], x_dimension=usedata.shape[1], learning_rate=1e-4,
				   BNTrainingMode=False)
	vae_eval.restore_model(path_vae_model_eval)

	starting_per_indices = list(pd.read_csv(os.path.join(path_start_per, "startingPerIndices.csv"))["Index"])
	targeting_per_indices = starting_per_indices.copy() + list(range(200))
	# (2) load models
	## lincs-vae
	# Tensors
	device = "cuda" if torch.cuda.is_available() else "cpu"

	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char=data_lincs_onehot.shape[2], max_len=data_lincs_onehot.shape[1]).to(device)
	model_chemvae.load_state_dict(torch.load(path_chemvae_model, map_location=device))
	model_chemvae.eval()

	# std model
	# std_model = Standardize(data_all = data_chem_onehot, model = model_chemvae, device = device)
	mu_std_model = np.load(os.path.join(path_std_param, "mu.npy"))
	std_std_model = np.load(os.path.join(path_std_param, "std.npy"))
	std_model = StandardizeLoad(mu_std_model, std_std_model, device)

	## cinn
	perturbToOnehot = {}
	for i in range(trt_list.shape[0]):
		perturbToOnehot[trt_list[i]] = i

	torch.manual_seed(42)
	flow_model = ConditionalFlatCouplingFlow(conditioning_dim=196,
											 # condition dimensions
											 embedding_dim=10,
											 conditioning_depth=2,
											 n_flows=20,
											 in_channels=10,
											 hidden_dim=1024,
											 hidden_depth=2,
											 activation="none",
											 conditioner_use_bn=True)

	model_c = Net2NetFlow_TFVAEFlow(configured_flow=flow_model,
									first_stage_data=usedata[idx_to_train][indicesWithOnehot][
														 kept_indices][:1000],
									cond_stage_data=perturb_with_onehot_kept[:1000],
									perturbToOnehotLib=perturbToOnehot,
									oneHotData=data_lincs_onehot,
									model_con=model_chemvae,
									std_model=std_model,
									sess=vae_train.sess,
									enc_ph=vae_train.x,
									z_gen_data_v=vae_train.z_mean,
									is_training=vae_train.is_training)

	### training
	model_c.to(device=device)
	model_c.train(n_epochs=1, batch_size=128, lr=4.5e-6)
	model_c.load(path_cinn_model)
	model_c.eval()

	model_g = model_c.model_con
	model_g.eval()

	perturbnet_model = TFVAEZ_CheckNet2Net(model_c, device, vae_train.sess, vae_train.x_hat, vae_train.z_mean,
										   vae_train.is_training)

	# (4) generating metric value
	Zsample = vae_train.encode(usedata)

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]

	pca_data_10 = PCA(n_components=10, random_state=42)
	pca_data_10_fit = pca_data_10.fit(Zsample)

	# baseline
	indices_vae_mb = np.random.choice(list(range(perturb_with_onehot_kept.shape[0])), n_per, replace = False)
	data_per_kept_sample = perturb_with_onehot_kept[indices_vae_mb]
	indices_onehot_kept_sample = []
	for per in data_per_kept_sample:
		indices_onehot_kept_sample.append(perturbToOnehot[per])
	indices_onehot_kept_sample = np.array(indices_onehot_kept_sample)

	for sce in range(len(list_string_interest)):
		scenario = list_string_interest[sce]
		trt_type = scenario

		## Input tensors
		# start B
		onehot_input = data_lincs_onehot[indices_onehot_kept_sample]
		_, _, _, embdata_torch_base = model_g(torch.tensor(onehot_input).float().to(device))
		onehot_input = onehot_input.reshape(onehot_input.shape[0], -1)

		## Z, Y
		trt_onehot_base = std_model.standardize_z(embdata_torch_base.cpu().detach().numpy())
		input_trt_latent_base_sample, _  = perturbnet_model.sample_data(trt_onehot_base)

		## B end
		indice_end = np.where(trt_obs_list == trt_type)[0][0]
		onehot_end = np.tile(data_lincs_onehot[indices_trt_kept][[indice_end]], (n_per, 1, 1))


		## Y end
		_, _, _, embdata_torch_end = model_g(torch.tensor(onehot_end).float().to(device))
		trt_onehot_end = std_model.standardize_z(embdata_torch_end.cpu().detach().numpy())
		onehot_end = onehot_end.reshape(onehot_end.shape[0], -1)


		# for Captum inputs y
		input_end_f = torch.tensor(np.concatenate((input_trt_latent_base_sample,
												   trt_onehot_base,
												   onehot_end), axis = 1))

		# for Captum inputs y
		input_start_f = torch.tensor(np.concatenate((input_trt_latent_base_sample,
													 trt_onehot_base,
													 onehot_input), axis = 1))


		input_end_y = torch.tensor(np.concatenate((input_trt_latent_base_sample,
												   trt_onehot_base,
												   trt_onehot_end), axis = 1))

		# for Captum inputs y
		input_start_y = torch.tensor(np.concatenate((input_trt_latent_base_sample,
													 trt_onehot_base,
													 trt_onehot_base), axis = 1))

		#for sce in range(len(compared_scenarios)):
		for c in [3, 10, 15, 19]:

			path_binaryClass_model = path_binaryClass_model_list + 'cluster_' + str(c)
			path_binaryClass_model = path_binaryClass_model + '/model_params_final_cluster' + str(c) + '_epoch_30.pt'

			model_class = BinaryCellStatesClass().to(device)
			model_class.load_state_dict(torch.load(path_binaryClass_model, map_location = device))
			model_class.eval()

			fModel_zLabel = FlowResizeLabelClass(model = model_c, model_g = model_g, model_class = model_class,
												 std_model = std_model, zDim = 10, yDim = 196,
												 n_seq = 120, n_vol = 35, device = device)

			yModel_zLabel = FlowResizeYLabelClass(model = model_c, model_g = model_g, model_class = model_class,
												  std_model = std_model, zDim = 10, yDim = 196,
												  n_seq = 120, n_vol = 35, device = device)


			ig_lab_f = IntegratedGradients(fModel_zLabel)
			ig_lab_y = IntegratedGradients(yModel_zLabel)

			d = 0
			score_start = ig_b_score_compute(ig = ig_lab_f,
											 input_data = input_end_f,
											 baseline_null = input_start_f,
											 target = d,
											 batch_size = 32,
											 ifPlot = True,
											 plot_save_file = os.path.join(path_save, 'b_keptDrug_' + str(indice_end) + '_cluster_' + str(c) + '_end_over_cinnKeptRandom_' + str(d) + '.png'))
			pd.DataFrame(score_start.copy()).to_csv(os.path.join(path_save, 'b_keptDrug_' + str(indice_end) + '_cluster_' + str(c) + '_end_over_cinnKeptRandom_' + str(d) + '.csv'))

			## general
			d = 0
			score_start = ig_y_score_compute(ig = ig_lab_y,
									 input_data = input_end_y,
									 baseline_null = input_start_y,
									 target = d,
									 batch_size = 32,
									 ifPlot = True,
									 plot_save_file = os.path.join(path_save, 'y_keptDrug_' + str(indice_end) + '_cluster_' + str(c) + '_end_over_cinnKeptRandom_' + str(d) + '.png'))

			pd.DataFrame(score_start.copy()).to_csv(os.path.join(path_save, 'y_keptDrug_' + str(indice_end) + '_cluster_' + str(c) + '_end_over_cinnKeptRandom_' + str(d) + '.csv'))
