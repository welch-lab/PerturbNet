#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../../..")
import os
import torch
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from perturbnet.perturb.util import *
from perturbnet.perturb.chemicalvae.chemicalVAE import *
from perturbnet.perturb.data_vae.modules.vae import *

if __name__ == "__main__":
	# (1) load data
	## directories
	path_save = "FIDDistances_LINCS"
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok=True)

	path_data = ""
	path_chemvae_model = ""
	path_vae_model_eval = ""
	path_lincs_onehot = os.path.join(path_data, "SmilesOneHot.npy")
	path_std_param = ""

	usedata = np.load(os.path.join(path_data, "data.npy"))

	## meta information
	input_ltpm_label = pd.read_csv(
		os.path.join(path_data, "PerturbMeta.csv"))
	perturb_with_onehot_overall = np.array(list(input_ltpm_label["canonical_smiles"]))

	## onehot
	data_lincs_onehot = np.load(path_lincs_onehot)
	trt_list = np.load(os.path.join(path_data, "Smiles.npy"))

	# evaluation vae
	vae = VAE(num_cells_train = usedata.shape[0], 
			  x_dimension = usedata.shape[1], 
			  learning_rate = 1e-4, 
			  BNTrainingMode = False)

	vae.restore_model(path_vae_model_eval)

	# (2) load models
	device = "cuda" if torch.cuda.is_available() else "cpu"

	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char = data_lincs_onehot.shape[2], 
								max_len = data_lincs_onehot.shape[1]).to(device)
	model_chemvae.load_state_dict(torch.load(path_chemvae_model, map_location=device))
	model_chemvae.eval()

	## standardization model
	mu_std_model = np.load(os.path.join(path_std_param, "mu.npy"))
	std_std_model = np.load(os.path.join(path_std_param, "std.npy"))
	std_model = StandardizeLoad(mu_std_model, std_std_model, device)

	## latent values of perturbations
	_, _, _, embdata_torch = model_chemvae(torch.tensor(data_lincs_onehot).float().to(device))
	embdata_numpy = std_model.standardize_z(embdata_torch.cpu().detach().numpy())

	neigh = NearestNeighbors(n_neighbors = 30)
	neigh_fit = neigh.fit(embdata_numpy)
	neigh_lib = neigh.kneighbors()[1].copy()

	# (3) metrics
	fidscore_vae_cal = fidscore_vae_extend(vae.sess, vae.z_mean, vae.mu, vae.x, vae.is_training)

	# (4) evaluations
	FID_matrix = np.zeros((len(trt_list), len(trt_list)))

	for i in range(len(trt_list)):
		perturb1 = trt_list[i]
		neigh_trts = neigh_lib[i]
		idx_trt_type1 = np.where(perturb_with_onehot_overall == perturb1)[0]

		for j in neigh_trts:

			perturb2 = trt_list[j]
			idx_trt_type2 = np.where(perturb_with_onehot_overall == perturb2)[0]

			fid_value, _ = fidscore_vae_cal.calculate_fid_vae_score(usedata[idx_trt_type1], 
																	usedata[idx_trt_type2], 
																	give_mean = True)

			if fid_value >= 0:
				FID_matrix[i, j] = fid_value

		if i % 20 == 10:
			np.save(os.path.join(path_save, "FID_30NN_MeanRep.npy"), FID_matrix)

	# transform FID distance matrix to Laplacian matrix
	FIDSyAvg = (FID_matrix + FID_matrix.T) / 2
	Lmat = np.exp( -FIDSyAvg)
	Lmat[Lmat == 1] = 0

	## normalized by row sum
	indices_with_rowSumZero = np.where(Lmat.sum(1) == 0)[0]
	Lmat = Lmat / Lmat.sum(axis = 1)[:, np.newaxis]

	for i in indices_with_rowSumZero:
		for j in range(Lmat.shape[1]):
			Lmat[i, j] = 0.0

	Lmat = np.eye(Lmat.shape[0]) - Lmat

	np.save(os.path.join(path_save, "Lmat_30NN_MeanRep.npy"), Lmat)

