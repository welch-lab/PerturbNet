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

from perturbnet.perturb.util import * 
from perturbnet.perturb.cinn.modules.flow import * 
from perturbnet.perturb.chemicalvae.chemicalVAE import *

if __name__ == "__main__":

	# (1) load data 
	## directories
	path_data = ""
	path_chemvae_model = ""
	path_scvi_model_cinn = ""

	path_cinn_model_save = "model"
	path_sciplex_onehot = ""
	path_chem_onehot = ""
	path_removed_per = ""
	path_std_param = ""

	## evalution scvi
	adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))	

	## sciplex trts
	trt_list = list(pd.read_csv(os.path.join(path_data, "trt.csv"))["treatment"])
	
	## onehot
	data_sciplex_onehot = np.load(path_sciplex_onehot)
	data_chem_onehot = np.load(path_chem_onehot)

	## meta information
	input_ltpm_label = adata.obs.copy()

	## undefined perturbations

	trt_cell_type_no = ["S0000", "nan"]
	list_c_trt = list(input_ltpm_label["treatment"])
	idx_to_train = [i for i in range(len(list_c_trt)) if list_c_trt[i] not in trt_cell_type_no]
	
	## removed perturbations
	perturb_with_onehot_overall = np.array(list(input_ltpm_label["treatment"]))
	input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]
	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train]
	
	removed_all_pers = np.load(os.path.join(path_removed_per, "RemovedPerturbs.npy"))

	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	input_ltpm_label1 = input_ltpm_label.iloc[kept_indices, :]
	input_ltpm_label1.index = list(range(input_ltpm_label1.shape[0]))

	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

	# perturbation information 
	data_trt = pd.read_csv(os.path.join(path_data, "trt.csv"))
	data_trt["Indices"] = list(range(data_trt.shape[0]))

	cell_embdata = input_ltpm_label1.loc[:, ["treatment"]].merge(data_trt, how = "left", on = "treatment")
	indices_onehot = list(cell_embdata["Indices"])

	data_sciplexKept_onehot = data_sciplex_onehot[indices_onehot]

	# (2) load models
	## generation scvi 
	adata_train = adata[idx_to_train, :].copy()
	adata_train = adata_train[kept_indices, :].copy()

	scvi.data.setup_anndata(adata_train, layer = "counts")
	scvi_model_cinn = scvi.model.SCVI.load(path_scvi_model_cinn, adata_train, use_cuda = False)
	scvi_model_de = scvi_predictive_z(scvi_model_cinn)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char = data_chem_onehot.shape[2], 
								max_len = data_chem_onehot.shape[1]).to(device)

	model_chemvae.load_state_dict(torch.load(path_chemvae_model, 
								  map_location = device))
	model_chemvae.eval()

	## standardization model
	mu_std_model = np.load(os.path.join(path_std_param, "mu.npy"))
	std_std_model = np.load(os.path.join(path_std_param, "std.npy"))
	std_model = StandardizeLoad(mu_std_model, std_std_model, device)


	## perturbnet
	## PCA 
	if sparse.issparse(adata.X):
		usedata = adata.X.A
	else:
		usedata = adata.X

	if sparse.issparse(adata.layers["counts"]):
		usedata_count = adata.layers["counts"].A
	else:
		usedata_count = adata.layers["counts"]


	flow_model = ConditionalFlatCouplingFlow(conditioning_dim = 196,
											 # condition dimensions 
											 embedding_dim = 10, 
											 conditioning_depth = 2, 
											 n_flows = 20, 
											 in_channels = 10, 
											 hidden_dim = 1024, 
											 hidden_depth = 2, 
											 activation = "none", 
											 conditioner_use_bn = True)


	model_c = Net2NetFlow_scVIChemStdFlow(configured_flow = flow_model,
										  first_stage_data = usedata_count[idx_to_train][kept_indices], 
										  cond_stage_data = data_sciplexKept_onehot, 
										  model_con = model_chemvae, 
										  scvi_model = scvi_model_cinn, 
										  std_model = std_model)


	model_c.to(device = device)
	model_c.train(n_epochs = 50, batch_size = 128, lr = 4.5e-6)
	#### save the model 
	model_c.save(path_cinn_model_save)