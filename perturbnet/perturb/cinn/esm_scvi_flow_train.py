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
from perturbnet.perturb.genotypevae.genotypeVAE import *


if __name__ == "__main__":

	# (1) load data
	## directories
	path_save = ""
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok = True)
	path_cinn_model_save =  ""

	path_data = ""
	path_scvi_model_train = ""

	a549_adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))

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
	scvi_model_train = scvi.model.SCVI.load(path_scvi_model_train, a549_adata_kept, use_cuda = False)

	# (2) load models
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	## esm
	data_meta = pd.read_csv(os.path.join(path_data, "sequence_representation.csv"))
	trt_list = np.array(list(data_meta['Variant']))
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
									  cond_stage_data = perturb_with_onehot_kept,
									  perturbToEmbedLib = perturbToEmbed,
									  embedData = embdata_numpy,
									  scvi_model = scvi_model_train)

	### training
	model_c.to(device = device)
	model_c.train_evaluateUnseenPer(anndata_unseen = a549_adata_removed,
									cond_stage_data_unseen = perturb_with_onehot_removed,
									path_save = path_save,
									n_epochs = 50, batch_size = 128, lr = 4.5e-6)

	#### save the model
	model_c.save(path_cinn_model_save)

