#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from genotypeVAE import *
import torch

if __name__ == "__main__":
	lambda_val = 1.0
	reload_model = False

	path_save = ""
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok=True)

	path_data = ""
	path_lincs_onehot = ""
	path_geno_onehot = ""

	path_genovae_model = ""
	path_fid = ""

	########################
	## Data preparation
	########################
	## onehot
	trt_list = np.load(os.path.join(path_data, "UniqueGenePerturbGene.npy"), allow_pickle = True)
	data_lincs_onehot = np.load(path_lincs_onehot)

	## meta information
	input_ltpm_label = pd.read_csv(
		os.path.join(path_data, "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_processed_PerturbMeta.csv"))
	idx_to_train = list(input_ltpm_label["pert_type"] == "trt_sh")
	perturb_with_onehot_overall = np.array(list(input_ltpm_label["pert_iname"]))
	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train]

	## with onehot
	data_perturb_set = set(trt_list)
	indicesWithOnehot = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] in data_perturb_set]

	perturb_with_onehot = perturb_with_onehot[indicesWithOnehot]

	## removed perturbations
	removed_all_pers = np.load(
		os.path.join(path_data + "onehot_geneticPerturbations", "LINCS_400RemovedGeneticPerturbs.npy"),
		allow_pickle=True)
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

	# library for perturbations
	perturbToOnehot = {}
	for i in range(trt_list.shape[0]):
		perturbToOnehot[trt_list[i]] = i

	# L matrix
	Lmatrix = np.load(os.path.join(path_fid, "Lmat_30NN_MeanRep.npy"))
	data_geno_onehot = load_npz(path_geno_onehot).toarray()

	##################
	# Model
	##################
	# train_loader = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True)
	# test_loader = torch.utils.data.DataLoader(data_test, batch_size=128, shuffle=True)
	torch.manual_seed(42)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = GenotypeVAE().to(device)
	if reload_model:
		model.load_state_dict(torch.load(path_genovae_model, map_location=device))

	model_zlz = GenotypeVAEZLZFineTune(genovae_model=model, Lmatrix=Lmatrix,
									   perturbToOnehot=perturbToOnehot, data_tune_onehot=data_lincs_onehot,
									   device=device)

	model_zlz.train_np(epochs=20, data_vae_onehot=data_geno_onehot, perturb_with_onehot=perturb_with_onehot_kept,
					   lambda_zlz=lambda_val, model_save_per_epochs=4, path_save=path_save)
