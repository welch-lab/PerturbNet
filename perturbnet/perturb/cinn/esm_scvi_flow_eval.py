#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../..')
import time
import os
import random

import torch
import torch.nn.functional as F
import torch.nn as nn

import anndata as ad
import scvi
from sklearn.decomposition import PCA

from pytorch_scvi.distributions import *
from pytorch_scvi.scvi_generate_z import *

from perturbnet.perturb.cinn.modules.flow import *
from perturbnet.perturb.cinn.modules.flow_generate import SCVIZ_CheckNet2Net

if __name__ == "__main__":

	# (1) load data
	## directories
	path_save = "output"
	path_cinn_model = ""

	path_data = ""
	path_scvi_model_train = ""
	path_scvi_model_eval = ""

	path_norm_param = ""

	a549_adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))
	scvi.data.setup_anndata(a549_adata, layer = "counts")
	scvi_model_eval = scvi.model.SCVI.load(path_scvi_model_eval, a549_adata, use_cuda=False)

	obs_names_kras_vardata = pd.read_csv(os.path.join(path_data, 'variants_library_kras_variant_seq.csv'))
	obs_names_tp53_vardata = pd.read_csv(os.path.join(path_data, 'variants_library_tp53_variant_seq.csv'))

	n_kras = obs_names_kras_vardata.shape[0]
	n_tp53 = obs_names_tp53_vardata.shape[0]

	idx_yes_kras = np.where(np.array(list(obs_names_kras_vardata['variant_seq_use'] == 'Yes')))[0]
	idx_yes_tp53 = np.where(np.array(list(obs_names_tp53_vardata['variant_seq_use'] == 'Yes')))[0]

	perturb_kras = np.array(list(obs_names_kras_vardata['variant_seq']))[idx_yes_kras]
	perturb_tp53 = np.array(list(obs_names_tp53_vardata['variant_seq']))[idx_yes_tp53]
	perturb_with_onehot_overall = np.concatenate((perturb_kras, perturb_tp53))

	removed_kras = np.load(os.path.join(path_data, 'removed_per_kras_60.npy'), allow_pickle=True)
	removed_tp53 = np.load(os.path.join(path_data, 'removed_per_tp53_70.npy'), allow_pickle=True)

	kept_kras_indices = [i for i in range(len(perturb_kras)) if perturb_kras[i] not in removed_kras]
	kept_tp53_indices = [i for i in range(len(perturb_tp53)) if perturb_tp53[i] not in removed_tp53]

	removed_kras_indices = np.setdiff1d(list(range(len(perturb_kras))),  kept_kras_indices)
	removed_tp53_indices = np.setdiff1d(list(range(len(perturb_tp53))),  kept_tp53_indices)


	perturb_with_onehot_kept = np.concatenate((np.array(perturb_kras)[kept_kras_indices],
											   np.array(perturb_tp53)[kept_tp53_indices]))
	perturb_with_onehot_removed = np.concatenate((np.array(perturb_kras)[removed_kras_indices],
												  np.array(perturb_tp53)[removed_tp53_indices]))

	idx_yes_tp53_use = np.array(list(range(len(range(n_tp53)))))[idx_yes_tp53]
	idx_yes_tp53_use_adjust = np.array([i + n_kras for i in idx_yes_tp53_use])

	idx_yes_tp53_kept = np.array(list(range(len(range(n_tp53)))))[idx_yes_tp53][kept_tp53_indices]
	idx_yes_tp53_kept_adjust = np.array([i + n_kras for i in idx_yes_tp53_kept])

	idx_yes_tp53_removed = np.array(list(range(len(range(n_tp53)))))[idx_yes_tp53][removed_tp53_indices]
	idx_yes_tp53_removed_adjust = np.array([i + n_kras for i in idx_yes_tp53_removed])

	idx_yes_kras_use_adjust = np.array(list(range(len(range(n_kras)))))[idx_yes_kras]
	idx_yes_kras_kept_adjust = np.array(list(range(len(range(n_kras)))))[idx_yes_kras][kept_kras_indices]
	idx_yes_kras_removed_adjust = np.array(list(range(len(range(n_kras)))))[idx_yes_kras][removed_kras_indices]

	idx_yes_kept = np.concatenate((idx_yes_kras_kept_adjust, idx_yes_tp53_kept_adjust))
	idx_yes_removed = np.concatenate((idx_yes_kras_removed_adjust, idx_yes_tp53_removed_adjust))
	idx_yes_use = np.concatenate((idx_yes_kras_use_adjust, idx_yes_tp53_use_adjust))

	a549_adata_use = a549_adata[idx_yes_use, :].copy()
	a549_adata_kept = a549_adata[idx_yes_kept, :].copy()
	a549_adata_removed = a549_adata[idx_yes_removed, :].copy()

	scvi.data.setup_anndata(a549_adata_kept, layer = "counts")
	scvi_model_cinn = scvi.model.SCVI.load(path_scvi_model_train, a549_adata_kept, use_cuda = False)
	scvi_model_de = scvi_predictive_z(scvi_model_cinn)

	n_subsample = 1000
	# (2) load models
	device = "cuda" if torch.cuda.is_available() else "cpu"

	## esm
	data_meta = pd.read_csv(os.path.join(path_data, "sequence_representation.csv"))
	trt_list = np.array(list(data_meta["Variant"]))
	removed_all_pers = np.concatenate((removed_kras, removed_tp53))

	perturbToEmbed = {}
	for i in range(trt_list.shape[0]):
		perturbToEmbed[trt_list[i]] = i

	embdata_numpy = data_meta.iloc[:, 1:-2].values

	torch.manual_seed(42)
	flow_model = ConditionalFlatCouplingFlow(conditioning_dim=1280,
											 # condition dimensions
											 embedding_dim=10,
											 conditioning_depth=2,
											 n_flows=20,
											 in_channels=10,
											 hidden_dim=1024,
											 hidden_depth=2,
											 activation="none",
											 conditioner_use_bn=True)

	model_c = Net2NetFlow_scVIFixFlow(configured_flow = flow_model,
									  cond_stage_data = perturb_with_onehot_kept[:n_subsample],
									  perturbToEmbedLib = perturbToEmbed,
									  embedData = embdata_numpy,
									  scvi_model = scvi_model_cinn)

	### training
	model_c.to(device = device)
	model_c.train_evaluateUnseenPer(anndata_unseen = a549_adata_removed[:n_subsample, :].copy(),
									cond_stage_data_unseen = perturb_with_onehot_removed[:n_subsample],
									n_epochs = 1, batch_size = 128, lr = 4.5e-6)

	#### save the model
	#### save the model
	model_c.load(path_cinn_model)
	model_c.eval()

	perturbnet_model = SCVIZ_CheckNet2Net(model_c, device, scvi_model_de)


	# (3) metrics
	## PCA
	if sparse.issparse(a549_adata_use.X):
		data_cell = a549_adata_use.X.A
	else:
		data_cell = a549_adata_use.X

	if sparse.issparse(a549_adata_use.layers['counts']):
		data_cell_cnt = a549_adata_use.layers['counts'].A
	else:
		data_cell_cnt = a549_adata_use.layers['counts']

	embdata_numpy = data_meta.iloc[:, 1:-2].values
	pca_data_50 = PCA(n_components=50, random_state = 42)
	pca_data_fit = pca_data_50.fit(data_cell)

	fidscore_cal = fidscore()
	RFE = RandomForestError()
	fidscore_scvi_cal = fidscore_scvi_extend(scvi_model = scvi_model_eval)

	normModel = NormalizedRevisionRSquare(largeCountData = data_cell_cnt)
	normModelVar = NormalizedRevisionRSquareVar(norm_model=normModel)

	# (4) evaluation
	Zsample = scvi_model_cinn.get_latent_representation(adata = a549_adata_use, give_mean = False)
	LSample = scvi_model_cinn.get_latent_library_size(adata = a549_adata_use, give_mean = False)

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]
	embdata_obs = embdata_numpy[indices_trt_kept]
	embdata_unseen = embdata_numpy[indices_trt_removed]

	save_results = SaveEvaluationResults("PerturbNet_Recon", "PerturbNet_Sample")


	## unseen perturbation output tables
	for indice_trt in range(len(trt_unseen_list)):

		trt_type = trt_unseen_list[indice_trt]


		## Method1
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]


		## PerturbNet
		onehot_indice_trt = np.tile(embdata_unseen[[indice_trt]], (len(idx_trt_type), 1))
		embdata_np = onehot_indice_trt + np.random.normal(scale = 0.001, size = onehot_indice_trt.shape)

		## recon data
		input_trt_latent, trt_onehot = Zsample[idx_trt_type], embdata_np
		library_trt_latent = LSample[idx_trt_type]

		_, fake_data = perturbnet_model.recon_data(input_trt_latent, trt_onehot, library_trt_latent)

		real_data = data_cell_cnt[idx_trt_type]

		### evaluation
		r2_value_d, real_norm, fake_norm = normModel.calculate_r_square(real_data, fake_data)

		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_norm, fake_norm, pca_data_fit, if_dataPC = True)
		errors_d = RFE.fit_once(real_norm, fake_norm, pca_data_fit, if_dataPC = True, output_AUC = False)

		fid_value_d_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean = False)
		fid_value_d_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean = True)


		## sampled data
		_, rfake_data = perturbnet_model.sample_data(trt_onehot, library_trt_latent)

		## evaluation
		r2_value_r, real_norm, rfake_norm = normModel.calculate_r_square(real_data, rfake_data)
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_norm, rfake_norm, pca_data_fit, if_dataPC = True)
		errors_r = RFE.fit_once(real_norm, rfake_norm, pca_data_fit, if_dataPC = True, output_AUC = False)

		fid_value_r_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean = False)
		fid_value_r_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean = True)

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

		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]


		## Method1
		onehot_indice_trt = np.tile(embdata_obs[[indice_trt]], (len(idx_trt_type), 1))
		embdata_np = onehot_indice_trt + np.random.normal(scale = 0.001, size = onehot_indice_trt.shape)

		## recon data
		input_trt_latent, trt_onehot = Zsample[idx_trt_type], embdata_np
		library_trt_latent = LSample[idx_trt_type]

		_, fake_data = perturbnet_model.recon_data(input_trt_latent, trt_onehot, library_trt_latent)

		real_data = data_cell_cnt[idx_trt_type]


		### evaluation
		r2_value_d, real_norm, fake_norm = normModel.calculate_r_square(real_data, fake_data)

		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_norm, fake_norm, pca_data_fit, if_dataPC = True)
		errors_d = RFE.fit_once(real_norm, fake_norm, pca_data_fit, if_dataPC = True, output_AUC = False)

		fid_value_d_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean = False)
		fid_value_d_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean = True)

		## sampled data
		_, rfake_data = perturbnet_model.sample_data(trt_onehot, library_trt_latent)

		## evaluation
		r2_value_r, real_norm, rfake_norm = normModel.calculate_r_square(real_data, rfake_data)
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_norm, rfake_norm, pca_data_fit, if_dataPC = True)
		errors_r = RFE.fit_once(real_norm, rfake_norm, pca_data_fit, if_dataPC = True, output_AUC = False)

		fid_value_r_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean = False)
		fid_value_r_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean = True)

		save_results.update(trt_type, len(idx_trt_type),
							r2_value_d, r2_value_r,
							fid_value_d, fid_value_r,
							errors_d, errors_r,
							fid_value_d_scvi_sample, fid_value_r_scvi_sample,
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save = path_save, file_save = "Observed", indice_start = len(trt_unseen_list))


