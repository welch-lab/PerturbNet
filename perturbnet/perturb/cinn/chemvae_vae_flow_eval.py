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

from perturbnet.perturb.knn.baseline_vae_knn import * 
from perturbnet.perturb.cinn.modules.flow_generate import TFVAEZ_CheckNet2Net


if __name__ == "__main__":

	# (1) load data 
	## directories
	path_save = "output"
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok = True)

	path_cinn_model = "model"
	path_data = ""
	path_chemvae_model = ""
	
	path_vae_model_eval = ""
	path_vae_model_train = ""
	
	path_lincs_onehot = ""
	path_chem_onehot = ""
	path_std_param = ""

	usedata = np.load(os.path.join(path_data, "data.npy"))
	
	## trts
	trt_list = np.load(os.path.join(path_lincs_onehot, "Smiles.npy"))
	
	## onehot
	data_lincs_onehot = np.load(os.path.join(path_lincs_onehot, "SmilesOneHot.npy"))
	data_chem_onehot = np.load(path_chem_onehot)

	## meta information 
	idx_to_train = np.load(os.path.join(path_lincs_onehot, "idx.npy"))
	input_ltpm_label = pd.read_csv(os.path.join(path_data, "PerturbMeta.csv"))

	perturb_with_onehot_overall = np.array(list(input_ltpm_label["canonical_smiles"]))
	input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]
	
	list_canonSmiles = list(input_ltpm_label['canonical_smiles'])
	#indicesWithOnehot = [i for i in range(len(list_canonSmiles)) if list_canonSmiles[i] in set(trt_list)]
	indicesWithOnehot = np.in1d(list_canonSmiles, trt_list)

	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train][indicesWithOnehot]

	removed_all_pers = np.load(os.path.join(path_lincs_onehot, "RemovedPerturbs.npy"))
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	input_ltpm_label1 = input_ltpm_label.iloc[kept_indices, :]
	input_ltpm_label1.index = list(range(input_ltpm_label1.shape[0]))

	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]


	vae_train = VAE(num_cells_train = usedata.shape[0], x_dimension = usedata.shape[1], learning_rate = 1e-4, BNTrainingMode = False)
	vae_train.restore_model(path_vae_model_train)

	vae_eval = VAE(num_cells_train = usedata.shape[0], x_dimension = usedata.shape[1], learning_rate = 1e-4, BNTrainingMode = False)
	vae_eval.restore_model(path_vae_model_eval)

	# (2) load models
	## lincs-vae
	# Tensors
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char = data_chem_onehot.shape[2], max_len = data_chem_onehot.shape[1]).to(device)
	model_chemvae.load_state_dict(torch.load(path_chemvae_model, map_location = device))
	model_chemvae.eval()

	# std model 
	#std_model = Standardize(data_all = data_chem_onehot, model = model_chemvae, device = device)
	mu_std_model = np.load(os.path.join(path_std_param, "mu.npy"))
	std_std_model = np.load(os.path.join(path_std_param, "std.npy"))
	std_model = StandardizeLoad(mu_std_model, std_std_model, device)

	## cinn
	perturbToOnehot = {}
	for i in range(trt_list.shape[0]):
		perturbToOnehot[trt_list[i]] = i

	torch.manual_seed(42)
	flow_model = ConditionalFlatCouplingFlow(  conditioning_dim = 196,
											   # condition dimensions 
											   embedding_dim = 10, 
											   conditioning_depth = 2, 
											   n_flows = 20, 
											   in_channels = 10, 
											   hidden_dim = 1024, 
											   hidden_depth = 2, 
											   activation = "none", 
											   conditioner_use_bn = True)

	model_c = Net2NetFlow_TFVAEFlow(configured_flow = flow_model,
									first_stage_data = usedata[idx_to_train][indicesWithOnehot][kept_indices][:1000], 
									cond_stage_data = perturb_with_onehot_kept[:1000],
									perturbToOnehotLib = perturbToOnehot,
									oneHotData = data_lincs_onehot, 
									model_con = model_chemvae, 
									std_model = std_model, 
									sess = vae_train.sess, 
									enc_ph = vae_train.x, 
									z_gen_data_v = vae_train.z_mean, 
									is_training = vae_train.is_training)

	### training
	model_c.to(device = device)
	model_c.train(n_epochs = 1, batch_size = 128, lr = 4.5e-6)
	model_c.load(path_cinn_model)
	model_c.eval()

	model_g = model_c.model_con
	model_g.eval()

	perturbnet_model = TFVAEZ_CheckNet2Net(model_c, device, vae_train.sess, vae_train.x_hat, vae_train.z_mean, vae_train.is_training)

	# (3) metrics
	## PCA 
	pca_data_50 = PCA(n_components=50, random_state = 42)
	pca_data_fit = pca_data_50.fit(usedata)

	fidscore_cal = fidscore()
	RFE = RandomForestError()
	fidscore_vae_cal = fidscore_vae_extend(vae_eval.sess, vae_eval.z_mean, vae_eval.mu, vae_eval.x, vae_eval.is_training)

	# (4) generating metric values
	Zsample = vae_train.encode(usedata)
	
	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]
	save_results = SaveEvaluationResults("PerturbNet_Recon", "PerturbNet_Sample")

	## unseen perturbation output tables 
	for indice_trt in range(len(trt_unseen_list)):

		trt_type = trt_unseen_list[indice_trt]

		## Method1
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		if idx_trt_type.shape[0] > 1000:
			idx_trt_type = np.random.choice(idx_trt_type, 1000, replace = False)

		onehot_indice_trt = np.tile(data_lincs_onehot[indices_trt_removed][[indice_trt]], (len(idx_trt_type), 1, 1))
		_, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))
		embdata_np = std_model.standardize_z(embdata_torch.cpu().detach().numpy())

		real_data = usedata[idx_trt_type]
		input_trt_latent = Zsample[idx_trt_type] 
		_, fake_data = perturbnet_model.recon_data(input_trt_latent, embdata_np)

		## evaluation
		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_data, fake_data, pca_data_fit, if_dataPC = True)
		errors_d = RFE.fit_once(real_data, fake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_d = fidscore_cal.calculate_r_square(real_data, fake_data)

		fid_value_d_scvi_sample, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean = False)
		fid_value_d_scvi_mu, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean = True)

		## Random 
		_, rfake_data = perturbnet_model.sample_data(embdata_np)

		## evaluation
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_data, rfake_data, pca_data_fit, if_dataPC = True)
		errors_r = RFE.fit_once(real_data, rfake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_r = fidscore_cal.calculate_r_square(real_data, rfake_data)

		fid_value_r_scvi_sample, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean = False)
		fid_value_r_scvi_mu, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean = True)

		save_results.update(trt_type, len(idx_trt_type), 
							r2_value_d, r2_value_r, 
							fid_value_d, fid_value_r, 
							errors_d, errors_r, 
							fid_value_d_scvi_sample, fid_value_r_scvi_sample, 
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save = path_save, file_save = "Unseen")

	## observed perturbations 
	for indice_trt in range(len(trt_obs_list)):

		trt_type = trt_obs_list[indice_trt]

		## Method1
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		if idx_trt_type.shape[0] > 1000:
			idx_trt_type = np.random.choice(idx_trt_type, 1000, replace = False)

		onehot_indice_trt = np.tile(data_lincs_onehot[indices_trt_kept][[indice_trt]], (len(idx_trt_type), 1, 1))
		_, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))
		embdata_np = std_model.standardize_z(embdata_torch.cpu().detach().numpy())
		
		real_data = usedata[idx_trt_type]
		input_trt_latent = Zsample[idx_trt_type] 
		_, fake_data = perturbnet_model.recon_data(input_trt_latent, embdata_np)

		## evaluation
		fid_value_d, _  = fidscore_cal.calculate_fid_score(real_data, fake_data, pca_data_fit, if_dataPC = True)
		errors_d = RFE.fit_once(real_data, fake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_d = fidscore_cal.calculate_r_square(real_data, fake_data)

		fid_value_d_scvi_sample, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean = False)
		fid_value_d_scvi_mu, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean = True)

		## Random 
		_, rfake_data = perturbnet_model.sample_data(embdata_np)
		
		## evaluation
		fid_value_r, _  = fidscore_cal.calculate_fid_score(real_data, rfake_data, pca_data_fit, if_dataPC = True)
		errors_r = RFE.fit_once(real_data, rfake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_r = fidscore_cal.calculate_r_square(real_data, rfake_data)

		fid_value_r_scvi_sample, _  = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean = False)
		fid_value_r_scvi_mu, _  = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean = True)

		save_results.update(trt_type, len(idx_trt_type), 
							r2_value_d, r2_value_r, 
							fid_value_d, fid_value_r, 
							errors_d, errors_r, 
							fid_value_d_scvi_sample, fid_value_r_scvi_sample, 
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save = path_save, file_save = "Observed", indice_start = len(trt_unseen_list))



