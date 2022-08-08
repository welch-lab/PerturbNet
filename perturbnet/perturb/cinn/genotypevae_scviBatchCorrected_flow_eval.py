#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../..')
import os
import time
import random
import anndata as ad

import torch
import torch.nn.functional as F
import torch.nn as nn

import scvi
from pytorch_scvi.distributions import *
from pytorch_scvi.scvi_generate_z import *
from sklearn.decomposition import PCA

from perturbnet.perturb.util import *
from perturbnet.perturb.cinn.modules.flow import *
from perturbnet.perturb.genotypevae.genotypeVAE import *
from perturbnet.perturb.cinn.modules.flow_generate import SCVIZ_CheckNet2Net

if __name__ == "__main__":

	# (1) load data
	## directories
	path_save = ""
	path_cinn_model = ""
	path_data = ""

	path_genovae_model = ""

	path_scvi_model_eval = ""
	path_scvi_model_eval_bc = ""
	path_scvi_model_train = ""

	adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))
	scvi.data.setup_anndata(adata, layer = "counts")
	scvi_model_eval = scvi.model.SCVI.load(path_scvi_model_eval, adata, use_cuda=False)

	#scvi.data.setup_anndata(adata, layer="counts", categorical_covariate_keys = ['gem_group'])
	scvi.data.setup_anndata(adata, layer = "counts", 
							batch_key = "gem_group")
	scvi_model_eval_bc = scvi.model.SCVI.load(path_scvi_model_eval_bc, adata, use_cuda=False)

	## onehot
	trt_list = np.load(os.path.join(path_data, "PerturbGene.npy"), allow_pickle = True)
	## onehot
	data_genomewide_onehot = np.load(os.path.join(path_data, "Onehot.npy"))


	## meta information
	input_ltpm_label = adata.obs
	perturb_with_onehot_overall = np.array(list(input_ltpm_label["gene"]))

	## with onehot
	data_perturb_set = set(trt_list)
	indicesWithOnehot = np.isin(perturb_with_onehot_overall, trt_list)
	#[i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] in data_perturb_set]

	perturb_with_onehot = perturb_with_onehot_overall[indicesWithOnehot]

	removed_all_pers = np.load(os.path.join(path_data, "RemovedPerturbs.npy"),
		allow_pickle = True)
	removed_all_pers_set = set(removed_all_pers)
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers_set]
	removed_indices = np.setdiff1d(list(range(len(perturb_with_onehot))), kept_indices)
	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]
	perturb_with_onehot_removed = perturb_with_onehot[removed_indices]

	# (2) load models
	scvi_model_cinn = scvi.model.SCVI.load(path_scvi_model_train, adata, use_cuda = False)
	scvi_model_de = scvi_predictive_z(scvi_model_cinn)

	n_subsample = 1000
	device = "cuda" if torch.cuda.is_available() else "cpu"

	## GenotypeVAE
	model_genovae = GenotypeVAE().to(device)
	model_genovae.load_state_dict(torch.load(path_genovae_model, map_location = device))
	model_genovae.eval()

	## cinn
	perturbToOnehot = {}
	for i in range(trt_list.shape[0]):
		perturbToOnehot[trt_list[i]] = i

	torch.manual_seed(42)
	flow_model = ConditionalFlatCouplingFlow(conditioning_dim=10,
											 # condition dimensions
											 embedding_dim=10,
											 conditioning_depth=2,
											 n_flows=20,
											 in_channels=10,
											 hidden_dim=1024,
											 hidden_depth=2,
											 activation="none",
											 conditioner_use_bn=True)

	model_c = Net2NetFlow_scVIGenoPerLibFlow(configured_flow=flow_model,
											 cond_stage_data=perturb_with_onehot_kept[:n_subsample],
											 perturbToOnehotLib = perturbToOnehot,
											 oneHotData = data_genomewide_onehot,
											 model_con=model_genovae,
											 scvi_model = scvi_model_cinn)

	### training
	model_c.to(device=device)
	model_c.train_evaluateUnseenPer(anndata_unseen = adata[:n_subsample, :].copy(),
									cond_stage_data_unseen = perturb_with_onehot_removed[:n_subsample],
									path_save = None,
									n_epochs = 1, batch_size = 128, lr = 4.5e-6)
	#### load the model
	model_c.load(path_cinn_model)
	model_c.eval()

	model_g = model_c.model_con
	model_g.eval()

	perturbnet_model = SCVIZ_CheckNet2Net(model_c, device, scvi_model_de)

	# (3) metrics
	if sparse.issparse(adata.X):
		usedata = adata.X.A
	else:
		usedata = adata.X

	if sparse.issparse(adata.layers["counts"]):
		usedata_count = adata.layers["counts"].A
	else:
		usedata_count = adata.layers["counts"]

	## PCA
	pca_data_50 = PCA(n_components=50, random_state=42)
	pca_data_fit = pca_data_50.fit(usedata)

	fidscore_cal = fidscore()
	RFE = RandomForestError()
	fidscore_scvi_cal = fidscore_scvi_extend(scvi_model = scvi_model_eval)
	fidscore_scvi_bc_cal = fidscore_scvi_extend(scvi_model = scvi_model_eval_bc)

	normModel = NormalizedRevisionRSquare(largeCountData = usedata_count)
	normModelVar = NormalizedRevisionRSquareVar(norm_model=normModel)

	# (4) generating metric values
	Zsample = scvi_model_cinn.get_latent_representation(adata = adata, give_mean = False)
	LSample = scvi_model_cinn.get_latent_library_size(adata = adata, give_mean = False)
	YSample = np.array([np.array(list(adata.obs['gem_group']))]).reshape(-1)
	BatchSample = np.array([np.array(list(adata.obs['_scvi_batch']))]).reshape(-1)

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = np.setdiff1d(list(range(len(trt_list))), indices_trt_removed)

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]
	save_results = SaveEvaluationResults("PerturbNet_Recon", "PerturbNet_Sample")

	## unseen perturbation output tables
	for indice_trt in range(len(trt_unseen_list)):

		trt_type = trt_unseen_list[indice_trt]

		## Method1
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		onehot_indice_trt = np.tile(data_genomewide_onehot[indices_trt_removed][[indice_trt]], (len(idx_trt_type), 1))
		_, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))
		embdata_np = embdata_torch.cpu().detach().numpy()

		real_data = usedata_count[idx_trt_type]
		input_trt_latent = Zsample[idx_trt_type]
		library_trt_latent = LSample[idx_trt_type]
		gem_latent = YSample[idx_trt_type]
		batch_latent = BatchSample[idx_trt_type]
		trt_onehot = embdata_np

		_, fake_data = perturbnet_model.recon_data_with_batch(input_trt_latent, trt_onehot, library_trt_latent, batch_latent)

		## evaluation
		r2_value_d, real_norm, fake_norm = normModel.calculate_r_square(real_data, fake_data)
		fid_value_d_scvi_mu_bc, _ = fidscore_scvi_bc_cal.calculate_fid_scvi_score_with_y(real_data, fake_data, gem_latent, 'gem_group', scvi, adata, give_mean = True)


		## Random
		_, rfake_data = perturbnet_model.sample_data_with_batch(trt_onehot, library_trt_latent, batch_latent)

		## evaluation
		r2_value_r, real_norm, rfake_norm = normModel.calculate_r_square(real_data, rfake_data)
		fid_value_r_scvi_mu_bc, _ = fidscore_scvi_bc_cal.calculate_fid_scvi_score_with_y(real_data, rfake_data, gem_latent, 'gem_group', scvi, adata, give_mean = True)


		save_results.update(trt_type, len(idx_trt_type),
							r2_value_d, r2_value_r,
							r2_value_d, r2_value_r,
							r2_value_d, r2_value_r,
							fid_value_d_scvi_mu_bc, fid_value_r_scvi_mu_bc,
							fid_value_d_scvi_mu_bc, fid_value_r_scvi_mu_bc)

		save_results.saveToCSV(path_save=path_save, file_save="Unseen")



	## observed perturbations
	for indice_trt in range(len(trt_obs_list)):

		trt_type = trt_obs_list[indice_trt]

		## Method1
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		onehot_indice_trt = np.tile(data_genomewide_onehot[indices_trt_kept][[indice_trt]], (len(idx_trt_type), 1))
		_, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))
		embdata_np = embdata_torch.cpu().detach().numpy()

		input_trt_latent, trt_onehot = Zsample[idx_trt_type], embdata_np
		library_trt_latent = LSample[idx_trt_type]
		gem_latent = YSample[idx_trt_type]
		batch_latent = BatchSample[idx_trt_type]

		_, fake_data = perturbnet_model.recon_data_with_batch(input_trt_latent, trt_onehot, library_trt_latent, batch_latent)

		real_data = usedata_count[idx_trt_type]


		## evaluation
		r2_value_d, real_norm, fake_norm = normModel.calculate_r_square(real_data, fake_data)
		fid_value_d_scvi_mu_bc, _ = fidscore_scvi_bc_cal.calculate_fid_scvi_score_with_y(real_data, fake_data, gem_latent, 'gem_group', scvi, adata, give_mean = True)

		## Random
		_, rfake_data = perturbnet_model.sample_data_with_batch(trt_onehot, library_trt_latent, batch_latent)

		## evaluation
		r2_value_r, real_norm, rfake_norm = normModel.calculate_r_square(real_data, rfake_data)
		fid_value_r_scvi_mu_bc, _ = fidscore_scvi_bc_cal.calculate_fid_scvi_score_with_y(real_data, rfake_data, gem_latent, 'gem_group', scvi, adata, give_mean = True)

		save_results.update(trt_type, len(idx_trt_type),
							r2_value_d, r2_value_r,
							r2_value_d, r2_value_r,
							r2_value_d, r2_value_r,
							fid_value_d_scvi_mu_bc, fid_value_r_scvi_mu_bc,
							fid_value_d_scvi_mu_bc, fid_value_r_scvi_mu_bc)

		save_results.saveToCSV(path_save=path_save, file_save="Observed", indice_start=len(trt_unseen_list))

