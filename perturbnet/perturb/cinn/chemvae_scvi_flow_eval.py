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
from perturbnet.perturb.cinn.modules.flow_generate import SCVIZ_CheckNet2Net

if __name__ == "__main__":

	# (1) load data
	## directories
	path_save = "output"
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok=True)

	path_data = ""
	path_chemvae_model = ""
	path_scvi_model_eval = ""
	path_scvi_model_cinn = ""

	path_cinn_model = ""
	path_sciplex_onehot = ""
	path_chem_onehot = ""
	path_removed_per = ""
	path_std_param = ""
	path_norm_param = ""

	## evalution scvi
	adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))
	scvi.data.setup_anndata(adata, layer = "counts")
	scvi_model_eval = scvi.model.SCVI.load(path_scvi_model_eval, adata, use_cuda = False)

	## sciplex trts
	trt_list = list(pd.read_csv(os.path.join(path_data, "trt.csv"))["treatment"])

	## onehot
	data_sciplex_onehot = np.load(path_sciplex_onehot)
	data_chem_onehot = np.load(path_chem_onehot)

	## meta information
	input_ltpm_label = adata.obs.copy()

	## removed perturbations
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
	scvi_model_cinn = scvi.model.SCVI.load(path_scvi_model_cinn, adata_train, use_cuda=False)
	scvi_model_de = scvi_predictive_z(scvi_model_cinn)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char = data_chem_onehot.shape[2], max_len = data_chem_onehot.shape[1]).to(device)
	model_chemvae.load_state_dict(torch.load(path_chemvae_model, map_location = device))
	model_chemvae.eval()

	## standardization model
	# std_model = Standardize(data_all=data_chem_onehot, model=model_chemvae, device=device)
	# np.save(os.path.join(path_std_param, 'mu.npy'), std_model.mu)
	# np.save(os.path.join(path_std_param, 'std.npy'), std_model.std)

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

	flow_model = ConditionalFlatCouplingFlow(conditioning_dim=196,
											 # condition dimensions
											 embedding_dim=10,
											 conditioning_depth=2,
											 n_flows=20,
											 in_channels=10,
											 hidden_dim=1024,
											 hidden_depth=2,
											 activation="none",
											 conditioner_use_bn=True)

	model_c = Net2NetFlow_scVIChemStdFlow(configured_flow=flow_model,
										  first_stage_data=usedata_count[idx_to_train][kept_indices][:1000],
										  cond_stage_data=data_sciplexKept_onehot[:1000],
										  model_con=model_chemvae,
										  scvi_model=scvi_model_cinn,
										  std_model=std_model)

	model_c.to(device = device)
	model_c.train(n_epochs = 1, batch_size = 128, lr = 4.5e-6)
	model_c.load(path_cinn_model)
	model_c.eval()

	model_g = model_c.model_con
	model_g.eval()

	perturbnet_model = SCVIZ_CheckNet2Net(model_c, device, scvi_model_de)

	# (3) metrics
	pca_data_50 = PCA(n_components=50, random_state=42)
	pca_data_fit = pca_data_50.fit(usedata)

	fidscore_cal = fidscore()
	RFE = RandomForestError()
	fidscore_scvi_cal = fidscore_scvi_extend(scvi_model=scvi_model_eval)
	# normModel = NormalizedRevisionRSquare(largeCountData=usedata_count)
	# np.save(os.path.join(path_norm_param, 'norm_mu.npy'), normModel.col_mu)
	# np.save(os.path.join(path_norm_param, 'norm_std.npy'), normModel.col_std)

	col_mu = np.load(os.path.join(path_norm_param, "norm_mu.npy"))
	col_std = np.load(os.path.join(path_norm_param, "norm_std.npy"))
	normModel = NormalizedRevisionRSquareLoad(col_mu = col_mu, col_std = col_std)
	normModelVar = NormalizedRevisionRSquareVar(norm_model = normModel)

	# (4) evaluation
	Zsample = scvi_model_cinn.get_latent_representation(adata = adata, give_mean = False)
	LSample = scvi_model_cinn.get_latent_library_size(adata = adata, give_mean = False)

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]

	save_results = SaveEvaluationResults("PerturbNet_Recon", "PerturbNet_Sample")

	## unseen perturbation output tables
	for indice_trt in range(len(trt_unseen_list)):
		trt_type = trt_unseen_list[indice_trt]
		## PerturbNet

		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		onehot_indice_trt = np.tile(data_sciplex_onehot[indices_trt_removed][[indice_trt]], (len(idx_trt_type), 1, 1))
		_, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))

		## recon data
		input_trt_latent, trt_onehot = Zsample[idx_trt_type], std_model.standardize_z(
			embdata_torch.cpu().detach().numpy())
		library_trt_latent = LSample[idx_trt_type]

		_, fake_data = perturbnet_model.recon_data(input_trt_latent, trt_onehot, library_trt_latent)

		real_data = usedata_count[idx_trt_type]

		### evaluation
		r2_value_d, real_norm, fake_norm = normModel.calculate_r_square(real_data, fake_data)

		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_norm, fake_norm, pca_data_fit, if_dataPC = True)
		errors_d = RFE.fit_once(real_norm, fake_norm, pca_data_fit, if_dataPC = True, output_AUC=False)

		fid_value_d_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean = False)
		fid_value_d_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean = True)


		## sampled data
		_, rfake_data = perturbnet_model.sample_data(trt_onehot, library_trt_latent)

		## evaluation
		r2_value_r, real_norm, rfake_norm = normModel.calculate_r_square(real_data, rfake_data)
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_norm, rfake_norm, pca_data_fit, if_dataPC=True)
		errors_r = RFE.fit_once(real_norm, rfake_norm, pca_data_fit, if_dataPC=True, output_AUC=False)

		fid_value_r_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean=False)
		fid_value_r_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean=True)

		
		save_results.update(trt_type, len(idx_trt_type),
							r2_value_d, r2_value_r,
							fid_value_d, fid_value_r,
							errors_d, errors_r,
							fid_value_d_scvi_sample, fid_value_r_scvi_sample,
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save=path_save, file_save = "Unseen")


	## observed perturbations
	for indice_trt in range(len(trt_obs_list)):
		trt_type = trt_obs_list[indice_trt]

		## Method1
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		onehot_indice_trt = np.tile(data_sciplex_onehot[indices_trt_kept][[indice_trt]], (len(idx_trt_type), 1, 1))
		_, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))

		## recon data
		input_trt_latent, trt_onehot = Zsample[idx_trt_type], std_model.standardize_z(
			embdata_torch.cpu().detach().numpy())
		library_trt_latent = LSample[idx_trt_type]

		_, fake_data = perturbnet_model.recon_data(input_trt_latent, trt_onehot, library_trt_latent)

		real_data = usedata_count[idx_trt_type]

		### evaluation
		r2_value_d, real_norm, fake_norm = normModel.calculate_r_square(real_data, fake_data)

		fid_value_d, _ = fidscore_cal.calculate_fid_score(real_norm, fake_norm, pca_data_fit, if_dataPC=True)
		errors_d = RFE.fit_once(real_norm, fake_norm, pca_data_fit, if_dataPC=True, output_AUC=False)

		fid_value_d_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean=False)
		fid_value_d_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, fake_data, give_mean=True)


		## sampled data
		_, rfake_data = perturbnet_model.sample_data(trt_onehot, library_trt_latent)

		## evaluation
		r2_value_r, real_norm, rfake_norm = normModel.calculate_r_square(real_data, rfake_data)
		fid_value_r, _ = fidscore_cal.calculate_fid_score(real_norm, rfake_norm, pca_data_fit, if_dataPC=True)
		errors_r = RFE.fit_once(real_norm, rfake_norm, pca_data_fit, if_dataPC=True, output_AUC=False)

		fid_value_r_scvi_sample, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean=False)
		fid_value_r_scvi_mu, _ = fidscore_scvi_cal.calculate_fid_scvi_score(real_data, rfake_data, give_mean=True)

		save_results.update(trt_type, len(idx_trt_type),
							r2_value_d, r2_value_r,
							fid_value_d, fid_value_r,
							errors_d, errors_r,
							fid_value_d_scvi_sample, fid_value_r_scvi_sample,
							fid_value_d_scvi_mu, fid_value_r_scvi_mu)

		save_results.saveToCSV(path_save=path_save, file_save="Observed", indice_start=len(trt_unseen_list))




