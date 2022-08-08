#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../../..")

import os
import random
import numpy as np
import pandas as pd
import scipy
import anndata as ad
import scvi
import torch

from scipy import stats, sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from perturbnet.perturb.util import *
from perturbnet.perturb.knn.knn import *
from perturbnet.perturb.chemicalvae.chemicalVAE import *

if __name__ == "__main__":

	# load data
	## directories

	path_save = "output"
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok = True)

	path_data = ""
	path_chemvae_model = ""
	path_scvi_model_eval = ""

	path_data_onehot = ""
	path_chem_onehot = ""
	path_removed_per = ""
	path_std_param = ""

	## load scvi
	adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))
	scvi.data.setup_anndata(adata, layer = "counts")
	scvi_model_eval = scvi.model.SCVI.load(path_scvi_model_eval, adata, use_cuda = False)

	## trts
	idx_to_train = ""
	trt_list = ""

	## onehot
	data_onehot = np.load(path_data_onehot)
	data_chem_onehot = np.load(path_chem_onehot)

	## meta information
	input_ltpm_label = adata.obs.copy()

	## removed perturbations
	perturb_with_onehot_overall = np.array(list(input_ltpm_label["treatment"]))
	input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]
	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train]

	removed_all_pers = np.load(os.path.join(path_removed_per, "RemovedPerturbs.npy"))

	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	input_ltpm_label1 = input_ltpm_label.iloc[kept_indices, :]

	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

	# load models
	device = "cuda" if torch.cuda.is_available() else "cpu"

	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char = data_chem_onehot.shape[2], 
								max_len = data_chem_onehot.shape[1]).to(device)
	model_chemvae.load_state_dict(torch.load(path_chemvae_model, map_location = device))
	model_chemvae.eval()

	# metrics
	## PCA
	if sparse.issparse(adata.X):
		usedata = adata.X.A
	else:
		usedata = adata.X

	if sparse.issparse(adata.layers["counts"]):
		usedata_count = adata.layers["counts"].A
	else:
		usedata_count = adata.layers["counts"]

	pca_data_50 = PCA(n_components=50, random_state=42)
	pca_data_fit = pca_data_50.fit(usedata)

	## standarize 
	mu_std_model = np.load(os.path.join(path_std_param, "mu.npy"))
	std_std_model = np.load(os.path.join(path_std_param, "std.npy"))
	std_model = StandardizeLoad(mu_std_model, std_std_model, device)

	## metrics' calculators
	fidscore_cal = fidscore()
	RFE = RandomForestError()
	fidscore_scvi_cal = fidscore_scvi_extend(scvi_model = scvi_model_eval)

	## latent values of perturbations
	_, _, _, embdata_torch = model_chemvae(torch.tensor(data_onehot).float().to(device))
	embdata_numpy = std_model.standardize_z(embdata_torch.cpu().detach().numpy())

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]
	embdata_obs = embdata_numpy[indices_trt_kept]
	embdata_unseen = embdata_numpy[indices_trt_removed]

	neigh = NearestNeighbors(n_neighbors=5)
	neigh_fit = neigh.fit(embdata_obs)

	save_results = SaveEvaluationResults("KNN", "Random")


	## unseen perturbation output tables
	for indice_trt in range(len(trt_unseen_list)):

		trt_type = trt_unseen_list[indice_trt]

		## KNN
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]
		idx_nontrt_type = np.array(list(range(len(perturb_with_onehot_kept))))

		distances, other_trts = neigh_fit.kneighbors(embdata_unseen[[indice_trt]], 5, return_distance=True)
		samplerNN = samplefromNeighbors(distances, other_trts)
		idx_sample = samplerNN.samplingTrt(trt_obs_list, perturb_with_onehot_kept, len(idx_trt_type))

		real_data, fake_data = usedata[idx_trt_type], usedata[idx_to_train][kept_indices][idx_sample]

		## evaluation
		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_data, fake_data, pca_data_fit, if_dataPC = True)
		errors_d = RFE.fit_once(real_data, fake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_d = fidscore_cal.calculate_r_square(real_data, fake_data)

		fid_value_d_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_sample], give_mean = False)
		fid_value_d_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_sample], give_mean = True)

		## Random
		idx_rsample = np.random.choice(idx_nontrt_type, len(idx_trt_type), replace=True)
		rfake_data = usedata[idx_to_train][kept_indices][idx_rsample]

		## evaluation
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_data, rfake_data, pca_data_fit, if_dataPC = True)
		errors_r = RFE.fit_once(real_data, rfake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_r = fidscore_cal.calculate_r_square(real_data, rfake_data)

		fid_value_r_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_rsample], give_mean = False)
		fid_value_r_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_rsample], give_mean = True)

		save_results.update(trt_type, len(idx_trt_type), 
							r2_value_d, r2_value_r, 
							fid_value_d, fid_value_r, 
							errors_d, errors_r, 
							fid_value_d_scvi_sample, fid_value_r_scvi_sample, 
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save = path_save, file_save = "Unseen")

	## observed perturbations
	distances_lib, neigh_lib = neigh_fit.kneighbors()[0].copy(), neigh_fit.kneighbors()[1].copy()
	for indice_trt in range(len(trt_obs_list)):
		trt_type = trt_obs_list[indice_trt]

		## KNN
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]
		idx_nontrt_type = np.where(perturb_with_onehot_kept != trt_type)[0]

		distances, other_trts = distances_lib[[indice_trt]], neigh_lib[[indice_trt]]
		samplerNN = samplefromNeighbors(distances, other_trts)
		idx_sample = samplerNN.samplingTrt(trt_obs_list, perturb_with_onehot_kept, len(idx_trt_type))

		real_data, fake_data = usedata[idx_trt_type], usedata[idx_to_train][kept_indices][idx_sample]

		## evaluation
		fid_value_d, _  = fidscore_cal.calculate_fid_score(real_data, fake_data, pca_data_fit, if_dataPC = True)
		errors_d = RFE.fit_once(real_data, fake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_d = fidscore_cal.calculate_r_square(real_data, fake_data)

		fid_value_d_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_sample], give_mean = False)
		fid_value_d_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_sample], give_mean = True)


		## Random
		idx_rsample = np.random.choice(idx_nontrt_type, len(idx_trt_type), replace=True)
		rfake_data = usedata[idx_to_train][kept_indices][idx_rsample]

		## evaluation
		fid_value_r, _  = fidscore_cal.calculate_fid_score(real_data, rfake_data, pca_data_fit, if_dataPC = True)
		errors_r = RFE.fit_once(real_data, rfake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
		r2_value_r = fidscore_cal.calculate_r_square(real_data, rfake_data)

		fid_value_r_scvi_sample, _  = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_rsample], give_mean = False)
		fid_value_r_scvi_mu, _  = fidscore_scvi_cal.calculate_fid_scvi_score(usedata_count[idx_trt_type], usedata_count[idx_to_train][kept_indices][idx_rsample], give_mean = True)

		save_results.update(trt_type, len(idx_trt_type), 
							r2_value_d, r2_value_r, 
							fid_value_d, fid_value_r, 
							errors_d, errors_r, 
							fid_value_d_scvi_sample, fid_value_r_scvi_sample, 
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save = path_save, file_save = "Observed", indice_start = len(trt_unseen_list))



