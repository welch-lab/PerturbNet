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

from pytorch_scvi.distributions import *
from pytorch_scvi.scvi_generate_z import *

import tensorflow as tf
from tensorflow import distributions as ds

from perturbnet.perturb.util import *
from perturbnet.perturb.cinn.modules.flow import *
from perturbnet.perturb.chemicalvae.chemicalVAE import *
from sklearn.neighbors import NearestNeighbors

from perturbnet.perturb.knn.baseline_knn_lincs import *
from perturbnet.perturb.cinn.modules.flow_generate import TFVAEZ_CheckNet2Net
from perturbnet.perturb.cinn.modules.flow_optim_trans import *

if __name__ == "__main__":

	# (1) load data
	## directories
	path_save = ""
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok=True)

	path_cinn_model = "model"
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

	_, _, _, embdata_torch = model_chemvae(torch.tensor(data_lincs_onehot).float().to(device))
	embdata_numpy = std_model.standardize_z(embdata_torch.cpu().detach().numpy())

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	embdata_obs = embdata_numpy[indices_trt_kept]
	embdata_unseen = embdata_numpy[indices_trt_removed]

	neigh_top = NearestNeighbors(n_neighbors = len(targeting_per_indices))
	neigh_top_fit = neigh_top.fit(embdata_obs[targeting_per_indices])

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]

	w_dis_cal = Wdistance()
	pd_knn_res = pd.DataFrame({'start': [], 'end': [], 'start_indice':[], 'trans_indice':[], 'fitted_indice':[],'knn_indice': []})
	n_smaller_distance_z = pd.DataFrame({'start': [], 'end': [], 'n': [], 'fitted': [], 'trans': [], 'fitted_w2': [], 'trans_w2': []})

	# the first/starting perturbation
	for indice_start in starting_per_indices:
		trt_type_base = trt_obs_list[indice_start]
		idx_trt_type_base = np.where(perturb_with_onehot_overall == trt_type_base)[0]

		onehot_indice_trt_base = np.tile(data_lincs_onehot[indices_trt_kept][[indice_start]],
										 (len(idx_trt_type_base), 1, 1))
		_, _, _, embdata_torch_base = model_g(torch.tensor(onehot_indice_trt_base).float().to(device))

		## recon data
		input_trt_latent_base, trt_onehot_base = Zsample[idx_trt_type_base], std_model.standardize_z(
			embdata_torch_base.cpu().detach().numpy())

		## observed perturbations
		for i in targeting_per_indices:
			trt_type = trt_obs_list[i]
			onehot_indice_trt_other = np.tile(data_lincs_onehot[indices_trt_kept][[i]],
											 (input_trt_latent_base.shape[0], 1, 1))
			_, _, _, embdata_torch_other = model_g(torch.tensor(onehot_indice_trt_other).float().to(device))
			trt_onehot_otherTo = std_model.standardize_z(embdata_torch_other.cpu().detach().numpy())
			fake_latent_other, _ = perturbnet_model.trans_data(input_trt_latent_base, trt_onehot_base,
															   trt_onehot_otherTo)
			np.save(os.path.join(path_save, "indice_" + str(indice_start) + "_trans_to_" + str(i) + ".npy"),
					fake_latent_other)
			if i % 10 == 0:
				print(i)

		for j in range(len(targeting_per_indices)):
			indice_trt = targeting_per_indices[j]
			trt_type = trt_obs_list[indice_trt]

			## Method1
			idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

			## recon data
			input_trt_latent = Zsample[idx_trt_type]

			# w2 distance of translated latent values to target
			W2_distance = []
			for i in targeting_per_indices:
				fake_latent_other = np.load(os.path.join(path_save, "indice_" + str(indice_start) + "_trans_to_" + str(i) + ".npy"))
				W2_distance.append(w_dis_cal.calculate_fid_score(fake_latent_other, input_trt_latent)[0])
			pd_res_knn = pd.DataFrame({'start': [indice_start] * len(W2_distance), 'target': targeting_per_indices, 'W2': W2_distance})
			pd_res_knn.to_csv(os.path.join(path_save, 'pd_res_' + "indice_" + str(indice_start) + '_trans_to_starting5AndFirst200' + '_w2dis.csv'))

			W2_distance = np.array(W2_distance)
			w2_val = np.min(W2_distance)
			indice_pred = np.argmin(W2_distance)
			indice_prediction = np.array(targeting_per_indices)[indice_pred]


			n_fitted = np.sum(w2_val <= W2_distance)
			n_trans = np.sum(W2_distance[j] <= W2_distance)

			n_smaller_distance_z = pd.concat(
				[n_smaller_distance_z, pd.DataFrame([[trt_type_base, trt_type, len(W2_distance), n_fitted, n_trans,  w2_val, W2_distance[j]]],
												 columns=['start', 'end', 'n', 'fitted', 'trans', 'fitted_w2', 'trans_w2'])])
			n_smaller_distance_z.to_csv(os.path.join(path_save, 'n_smaller_distance_z.csv'))

			# lincs
			newfig = plt.figure(figsize = (12, 9))
			plt.hist(W2_distance, bins=20, color='c', edgecolor='k', alpha=0.65)
			plt.axvline(W2_distance[j], color='k', linestyle='dashed', linewidth=1, label = 'Trans W2')
			plt.axvline(w2_val, color='red', linestyle='dashed', linewidth=1, label = 'Fitted W2')
			plt.title(trt_type + ': fittedSmaller: ' + str(n_fitted) + ', transSmaller: ' + str(n_trans))
			plt.legend()
			newfig.savefig(os.path.join(path_save, "indice_" + str(indice_start) + "_to_hist_lincs_" + str(indice_trt) +  ".png"))

			distances, other_trts = neigh_top_fit.kneighbors(embdata_obs[[indice_prediction]], return_distance=True)
			indice_knn = np.where(np.array(targeting_per_indices)[other_trts[0]] == indice_trt)[0]
			if len(indice_knn) == 0:
				indice_knn_fit = -1
			else:
				indice_knn_fit = indice_knn[0]

			pd_knn_res = pd.concat(
				[pd_knn_res, pd.DataFrame([[trt_type_base, trt_type, indice_start, indice_trt, indice_prediction, indice_knn_fit]],
										  columns=['start', 'end', 'start_indice','trans_indice', 'fitted_indice','knn_indice'])])
			pd_knn_res.to_csv(os.path.join(path_save, 'pd_knn_res.csv'))
