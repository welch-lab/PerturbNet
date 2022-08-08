#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import anndata as ad
import pandas as pd
import numpy as np
import scvi

if __name__ == "__main__":
	path_data = ""
	adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))
	scvi.data.setup_anndata(adata, layer = "counts", batch_key = "gem_group")
	## onehot
	trt_list = np.load(os.path.join(path_data, "PerturbGene.npy"), allow_pickle=True)
	data_lincs_onehot = np.load(os.path.join(path_data, "Onehot.npy"))

	## meta information
	input_ltpm_label = adata.obs
	perturb_with_onehot_overall = np.array(list(input_ltpm_label["gene"]))

	## with onehot
	data_perturb_set = set(trt_list)
	indicesWithOnehot = np.isin(perturb_with_onehot_overall, trt_list)
	#[i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] in data_perturb_set]
	indicesWithOnehot_idx = np.where(indicesWithOnehot)[0]
	perturb_with_onehot = perturb_with_onehot_overall[indicesWithOnehot]

	removed_all_pers = np.load(os.path.join(path_data, "RemovedPerturbs.npy"),
		allow_pickle = True)
	removed_all_pers_set = set(removed_all_pers)
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers_set]
	removed_indices = np.setdiff1d(list(range(len(perturb_with_onehot))), kept_indices)
	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

	adata = adata[indicesWithOnehot_idx, :].copy()
	adata = adata[kept_indices, :].copy()

	# train the model
	#scvi.data.setup_anndata(adata, layer = "counts", categorical_covariate_keys = ['gem_group'])
	#scvi.data.setup_anndata(adata, layer = "counts", batch_key = 'gem_group')
	model = scvi.model.SCVI(adata, n_latent = 10)
	model.train(n_epochs = 400)
	model.save("./model/")
