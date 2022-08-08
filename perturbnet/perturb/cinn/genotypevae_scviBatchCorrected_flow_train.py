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

import anndata as ad
import scvi
from pytorch_scvi.distributions import *
from pytorch_scvi.scvi_generate_z import *


from perturbnet.drug_perturb.util import *
from perturbnet.perturb.cinn.modules.flow import *
from perturbnet.perturb.genotypevae.genotypeVAE import *

if __name__ == "__main__":

	path_save = ""
	path_cinn_model_save = ""
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok = True)

	path_data = ""
	path_genovae_model = ""
	path_scvi_model_train = ""

	adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))
	## onehot
	trt_list = np.load(os.path.join(path_data, "PerturbGene.npy"), allow_pickle = True)
	## onehot
	data_genomewide_onehot = np.load(os.path.join(path_data, "Onehot.npy"))


	## meta information
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
	perturb_with_onehot_removed = perturb_with_onehot[removed_indices]


	# (2) load models
	## lincs-vae
	# Tensors
	scvi.data.setup_anndata(adata, layer = "counts", batch_key = 'gem_group')
	adata = adata[indicesWithOnehot_idx, :].copy()
	adataUnseen = adata[removed_indices, :].copy()
	adata = adata[kept_indices, :].copy()

	#scvi.data.setup_anndata(adata, layer = "counts", categorical_covariate_keys = ['gem_group'])

	scvi_model_train = scvi.model.SCVI.load(path_scvi_model_train, adata, use_cuda = False)
	#scvi.data.setup_anndata(adataUnseen, layer = "counts", categorical_covariate_keys = ['gem_group'])
	#scvi.data.setup_anndata(adataUnseen, layer = "counts", batch_key = 'gem_group')

	device = "cuda" if torch.cuda.is_available() else "cpu"

	## GenotypeVAE
	model_genovae = GenotypeVAE().to(device)
	model_genovae.load_state_dict(torch.load(path_genovae_model, map_location=device))
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
											 cond_stage_data=perturb_with_onehot_kept,
											 perturbToOnehotLib = perturbToOnehot,
											 oneHotData = data_genomewide_onehot,
											 model_con = model_genovae,
											 scvi_model = scvi_model_train)
	### training
	model_c.to(device=device)
	model_c.train_evaluateUnseenPer(anndata_unseen = adataUnseen,
									cond_stage_data_unseen = perturb_with_onehot_removed,
									path_save = path_save,
									n_epochs = 50, batch_size = 128, lr = 4.5e-6)
	#### save the model
	model_c.save(path_cinn_model_save)


