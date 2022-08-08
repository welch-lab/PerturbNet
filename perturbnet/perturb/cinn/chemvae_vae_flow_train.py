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

import tensorflow as tf
from tensorflow import distributions as ds

from perturbnet.perturb.util import * 
from perturbnet.perturb.cinn.modules.flow import * 
from perturbnet.perturb.chemicalvae.chemicalVAE import *
from perturbnet.perturb.knn.baseline_vae_knn import * 
from perturbnet.perturb.data_vae.modules.vae import *


if __name__ == "__main__":

	path_cinn_model_save = "model"
	path_data = ""
	path_chemvae_model = ""
	
	path_vae_model = ""
	path_lincs_onehot = ""
	path_chem_onehot = ""
	path_std_param = ""


	usedata = np.load(os.path.join(path_data, "data.npy"))
	
	## trts
	trt_list = np.load(os.path.join(path_lincs_onehot, "Smiles.npy"))
	
	## onehot
	data_lincs_onehot = np.load(os.path.join(path_lincs_onehot, "OneHot.npy"))
	data_chem_onehot = np.load(path_chem_onehot)

	## meta information 
	idx_to_train = np.load(os.path.join(path_lincs_onehot, "idx.npy"))
	input_ltpm_label = pd.read_csv(os.path.join(path_data, "PerturbMeta.csv"))

	perturb_with_onehot_overall = np.array(list(input_ltpm_label['canonical_smiles']))
	input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]
	

	list_canonSmiles = list(input_ltpm_label["canonical_smiles"])
	#indicesWithOnehot = [i for i in range(len(list_canonSmiles)) if list_canonSmiles[i] in set(trt_list)]
	indicesWithOnehot = np.in1d(list_canonSmiles, trt_list)
	
	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train][indicesWithOnehot]

	removed_all_pers = np.load(os.path.join(path_lincs_onehot, "RemovedPerturbs.npy"))
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]

	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

	# (2) load models
	## lincs-vae
	# Tensors
	vae = VAE(num_cells_train = usedata.shape[0], x_dimension = usedata.shape[1], learning_rate = 1e-4, BNTrainingMode = False)
	vae.restore_model(path_vae_model)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char = data_chem_onehot.shape[2], 
								max_len = data_chem_onehot.shape[1]).to(device)
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
									first_stage_data = usedata[idx_to_train][indicesWithOnehot][kept_indices], 
									cond_stage_data = perturb_with_onehot_kept,
									perturbToOnehotLib = perturbToOnehot,
									oneHotData = data_lincs_onehot, 
									model_con = model_chemvae, 
									std_model = std_model, 
									sess = vae.sess, 
									enc_ph = vae.x, 
									z_gen_data_v = vae.z_mean, 
									is_training = vae.is_training
									)

	### training
	model_c.to(device = device)
	model_c.train(n_epochs = 50, batch_size = 128, lr = 4.5e-6)
	#### save the model 
	model_c.save(path_cinn_model_save)



	


