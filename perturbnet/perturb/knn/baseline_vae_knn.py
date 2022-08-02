#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../..')
import os
import random
import time
import numpy as np
import pandas as pd
import scipy
from scipy import stats, sparse

import anndata as ad
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import torch

from perturbnet.perturb.util import *
from perturbnet.drug_perturb.knn.knn import *
from perturbnet.genetic_perturb.genotypevae.genotypeVAE import *
from perturbnet.drug_perturb.data_vae.modules.vae import *


if __name__ == "__main__":
	# (1) load data
	## directories
	path_save = "output"

	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok = True)

	path_data = ""
	path_genovae_model = ""
	path_vae_model_eval =  ""

	path_lincs_onehot =  ""
	usedata = np.load(os.path.join(path_data, "data.npy"))

	# evaluation vae
	vae = VAE(num_cells_train = usedata.shape[0], x_dimension = usedata.shape[1], learning_rate=1e-4, BNTrainingMode=False)
	vae.restore_model(path_vae_model_eval)

	## trts
	trt_list = ""

	## onehot
	data_lincs_onehot = np.load(path_lincs_onehot)

	## meta information
	input_ltpm_label = pd.read_csv(
		os.path.join(path_data, 'PerturbMeta.csv'))
	idx_to_train = ""
	perturb_with_onehot_overall = np.array(list(input_ltpm_label['pert_iname']))

	input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]
	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train]

	removed_all_pers = np.load(os.path.join(path_data + '/onehot_geneticPerturbations', "LINCS_400RemovedGeneticPerturbs.npy"), allow_pickle = True)
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	input_ltpm_label = input_ltpm_label.iloc[kept_indices, :]
	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

	# (2) load models
	device = "cuda" if torch.cuda.is_available() else "cpu"

	## GenotypeVAE
	model_genovae = GenotypeVAE().to(device)
	model_genovae.load_state_dict(torch.load(path_genovae_model, map_location=device))
	model_genovae.eval()

	# (3) metrics
	## PCA
	pca_data_50 = PCA(n_components=50, random_state=42)
	pca_data_fit = pca_data_50.fit(usedata)

	fidscore_cal = fidscore()
	RFE = RandomForestError()
	fidscore_vae_cal = fidscore_vae_extend(vae.sess, vae.z_mean, vae.mu, vae.x, vae.is_training)

	# (4) generating metric values
	_, _, _, embdata_torch = model_genovae(torch.tensor(data_lincs_onehot).float().to(device))
	embdata_numpy = embdata_torch.cpu().detach().numpy()

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]
	embdata_obs = embdata_numpy[indices_trt_kept]
	embdata_unseen = embdata_numpy[indices_trt_removed]

	neigh = NearestNeighbors(n_neighbors = 5)
	neigh_fit = neigh.fit(embdata_obs)

	save_results = SaveEvaluationResults("KNN", "Random")

	## unseen perturbation output tables
	for indice_trt in range(len(trt_unseen_list)):

		trt_type = trt_unseen_list[indice_trt]

		## KNN
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		if idx_trt_type.shape[0] > 1000:
			idx_trt_type = np.random.choice(idx_trt_type, 1000, replace = False)

		idx_nontrt_type = np.array(list(range(len(perturb_with_onehot_kept))))

		distances, other_trts = neigh_fit.kneighbors(embdata_unseen[[indice_trt]], 5, return_distance=True)
		samplerNN = samplefromNeighbors(distances, other_trts)
		idx_sample = samplerNN.samplingTrt(trt_obs_list, perturb_with_onehot_kept, len(idx_trt_type))

		real_data, fake_data = usedata[idx_trt_type], usedata[idx_to_train][kept_indices][idx_sample]

		## evaluation
		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_data, fake_data, pca_data_fit, if_dataPC=True)
		errors_d = RFE.fit_once(real_data, fake_data, pca_data_fit, if_dataPC=True, output_AUC=False)
		r2_value_d = fidscore_cal.calculate_r_square(real_data, fake_data)

		fid_value_d_scvi_sample, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean=False)
		fid_value_d_scvi_mu, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean=True)

		## Random
		idx_rsample = np.random.choice(idx_nontrt_type, len(idx_trt_type), replace=True)
		rfake_data = usedata[idx_to_train][kept_indices][idx_rsample]

		## evaluation
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_data, rfake_data, pca_data_fit, if_dataPC=True)
		errors_r = RFE.fit_once(real_data, rfake_data, pca_data_fit, if_dataPC=True, output_AUC=False)
		r2_value_r = fidscore_cal.calculate_r_square(real_data, rfake_data)

		fid_value_r_scvi_sample, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean=False)
		fid_value_r_scvi_mu, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean=True)

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

		if idx_trt_type.shape[0] > 1000:
			idx_trt_type = np.random.choice(idx_trt_type, 1000, replace=False)

		idx_nontrt_type = np.where(perturb_with_onehot_kept != trt_type)[0]

		distances, other_trts = distances_lib[[indice_trt]], neigh_lib[[indice_trt]]
		samplerNN = samplefromNeighbors(distances, other_trts)
		idx_sample = samplerNN.samplingTrt(trt_obs_list, perturb_with_onehot_kept, len(idx_trt_type))

		real_data, fake_data = usedata[idx_trt_type], usedata[idx_to_train][kept_indices][idx_sample]

		## evaluation
		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_data, fake_data, pca_data_fit, if_dataPC=True)
		errors_d = RFE.fit_once(real_data, fake_data, pca_data_fit, if_dataPC=True, output_AUC=False)
		r2_value_d = fidscore_cal.calculate_r_square(real_data, fake_data)

		fid_value_d_scvi_sample, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean=False)
		fid_value_d_scvi_mu, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, fake_data, give_mean=True)

		## Random
		idx_rsample = np.random.choice(idx_nontrt_type, len(idx_trt_type), replace=True)
		rfake_data = usedata[idx_to_train][kept_indices][idx_rsample]

		## evaluation
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_data, rfake_data, pca_data_fit, if_dataPC=True)
		errors_r = RFE.fit_once(real_data, rfake_data, pca_data_fit, if_dataPC=True, output_AUC=False)
		r2_value_r = fidscore_cal.calculate_r_square(real_data, rfake_data)

		fid_value_r_scvi_sample, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean=False)
		fid_value_r_scvi_mu, _ = fidscore_vae_cal.calculate_fid_vae_score(real_data, rfake_data, give_mean=True)

		save_results.update(trt_type, len(idx_trt_type),
							r2_value_d, r2_value_r,
							fid_value_d, fid_value_r,
							errors_d, errors_r,
							fid_value_d_scvi_sample, fid_value_r_scvi_sample,
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save=path_save, file_save="Observed", indice_start=len(trt_unseen_list))

