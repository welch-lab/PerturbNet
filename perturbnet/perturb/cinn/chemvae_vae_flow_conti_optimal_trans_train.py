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

from pytorch_scvi.distributions import *
from pytorch_scvi.scvi_generate_z import *

import tensorflow as tf
from tensorflow import distributions as ds

from perturbnet.perturb.util import *
from perturbnet.perturb.cinn.modules.flow import *
from perturbnet.perturb.chemicalvae.chemicalVAE import *
from sklearn.neighbors import NearestNeighbors

from perturbnet.perturb.knn.baseline_knn_lincs import *
from perturbnet.perturb.cinn.modules.flow_generate import TFVAEZ_CheckNet2Net
from perturbnet.perturb.cinn.modules.flow_optim_trans import *

if __name__ == "__main__":

	# (1) load data
	## directories
	path_save = ""
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok=True)

	path_cinn_model = ""
	path_data = ""
	path_chemvae_model = ""

	path_vae_model_eval = ""
	path_vae_model_train = ""

	path_lincs_onehot = ""
	path_chem_onehot = ""
	path_std_param = ""

	usedata = np.load(os.path.join(path_data, "data.npy"))

	## trts
	trt_list = np.load(os.path.join(path_lincs_onehot,
									"Smiles.npy"))

	## onehot
	data_lincs_onehot = np.load(os.path.join(path_lincs_onehot,
											 "SmilesOneHot.npy"))
	data_chem_onehot = np.load(path_chem_onehot)

	## meta information
	idx_to_train = np.load(os.path.join(path_lincs_onehot,
										"idx.npy"))
	input_ltpm_label = pd.read_csv(
		os.path.join(path_data,  "PerturbMeta.csv"))

	perturb_with_onehot_overall = np.array(list(input_ltpm_label["canonical_smiles"]))
	input_ltpm_label = input_ltpm_label.iloc[idx_to_train, :]


	list_canonSmiles = list(input_ltpm_label["canonical_smiles"])
	# indicesWithOnehot = [i for i in range(len(list_canonSmiles)) if list_canonSmiles[i] in set(trt_list)]
	indicesWithOnehot = np.in1d(list_canonSmiles, trt_list)

	perturb_with_onehot = perturb_with_onehot_overall[idx_to_train][indicesWithOnehot]

	removed_all_pers = np.load(os.path.join(path_lincs_onehot, "RemovedPerturbs.npy"))
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	input_ltpm_label1 = input_ltpm_label.iloc[kept_indices, :]
	input_ltpm_label1.index = list(range(input_ltpm_label1.shape[0]))

	perturb_with_onehot_kept = perturb_with_onehot[kept_indices]

	vae_train = VAE(num_cells_train=usedata.shape[0], x_dimension=usedata.shape[1], learning_rate=1e-4,
					BNTrainingMode=False)
	vae_train.restore_model(path_vae_model_train)

	vae_eval = VAE(num_cells_train=usedata.shape[0], x_dimension=usedata.shape[1], learning_rate=1e-4,
				   BNTrainingMode=False)
	vae_eval.restore_model(path_vae_model_eval)

	# (2) load models
	## lincs-vae
	# Tensors
	device = "cuda" if torch.cuda.is_available() else "cpu"

	## ChemicalVAE
	model_chemvae = ChemicalVAE(n_char=data_chem_onehot.shape[2], max_len=data_chem_onehot.shape[1]).to(device)
	model_chemvae.load_state_dict(torch.load(path_chemvae_model, map_location=device))
	model_chemvae.eval()

	# std model
	# std_model = Standardize(data_all = data_chem_onehot, model = model_chemvae, device = device)
	mu_std_model = np.load(os.path.join(path_std_param, "mu.npy"))
	std_std_model = np.load(os.path.join(path_std_param, "std.npy"))
	std_model = StandardizeLoad(mu_std_model, std_std_model, device)

	## cinn
	perturbToOnehot = {}
	for i in range(trt_list.shape[0]):
		perturbToOnehot[trt_list[i]] = i

	torch.manual_seed(42)
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

	model_c = Net2NetFlow_TFVAEFlow(configured_flow=flow_model,
									first_stage_data=usedata[idx_to_train][indicesWithOnehot][kept_indices][:1000],
									cond_stage_data=perturb_with_onehot_kept[:1000],
									perturbToOnehotLib=perturbToOnehot,
									oneHotData=data_lincs_onehot,
									model_con=model_chemvae,
									std_model=std_model,
									sess=vae_train.sess,
									enc_ph=vae_train.x,
									z_gen_data_v=vae_train.z_mean,
									is_training=vae_train.is_training)

	### training
	model_c.to(device=device)
	model_c.train(n_epochs=1, batch_size=128, lr=4.5e-6)
	model_c.load(path_cinn_model)
	model_c.eval()

	model_g = model_c.model_con
	model_g.eval()

	perturbnet_model = TFVAEZ_CheckNet2Net(model_c, device, vae_train.sess, vae_train.x_hat, vae_train.z_mean,
										   vae_train.is_training)


	# (4) generating metric value
	# _, _, _, embdata_torch = model_chemvae(torch.tensor(data_lincs_onehot).float().to(device))
	# embdata_numpy = std_model.standardize_z(embdata_torch.cpu().detach().numpy())

	# (4) generating metric values
	Zsample = vae_train.encode(usedata)

	indices_trt_removed = [i for i in range(len(trt_list)) if trt_list[i] in removed_all_pers]
	indices_trt_kept = [i for i in range(len(trt_list)) if i not in set(indices_trt_removed)]

	# embdata_obs = embdata_numpy[indices_trt_kept]
	# embdata_unseen = embdata_numpy[indices_trt_removed]

	# neigh = NearestNeighbors(n_neighbors=5)
	# neigh_fit = neigh.fit(embdata_obs)

	trt_obs_list, trt_unseen_list = np.array(trt_list)[indices_trt_kept], np.array(trt_list)[indices_trt_removed]

	# the first/starting perturbation
	indice_start = 0
	trt_type_base = trt_obs_list[indice_start]
	idx_trt_type_base = np.where(perturb_with_onehot_overall == trt_type_base)[0]

	if idx_trt_type_base.shape[0] > 1000:
		idx_trt_type_base = np.random.choice(idx_trt_type_base, 1000, replace=False)

	onehot_indice_trt_base = np.tile(data_lincs_onehot[indices_trt_kept][[indice_start]],
									 (len(idx_trt_type_base), 1, 1))
	_, _, _, embdata_torch_base = model_g(torch.tensor(onehot_indice_trt_base).float().to(device))

	## recon data
	input_trt_latent_base, trt_onehot_base = Zsample[idx_trt_type_base], std_model.standardize_z(
		embdata_torch_base.cpu().detach().numpy())

	## observed perturbations
	list_knn_indices = []
	list_res_w2dis = []
	pd_res_fittedW2Dis = pd.DataFrame({'start': [], 'target': [], 'W2': []})
	w_dis_cal = Wdistance()
	for indice_trt in range(0, 200):

		trt_type = trt_obs_list[indice_trt]

		## Method1
		idx_trt_type = np.where(perturb_with_onehot_overall == trt_type)[0]

		if idx_trt_type.shape[0] > 1000:
			idx_trt_type = np.random.choice(idx_trt_type, 1000, replace=False)

		onehot_indice_trt = np.tile(data_lincs_onehot[indices_trt_kept][[indice_trt]], (len(idx_trt_type), 1, 1))
		_, _, _, embdata_torch = model_g(torch.tensor(onehot_indice_trt).float().to(device))
		embdata_np = std_model.standardize_z(embdata_torch.cpu().detach().numpy())

		## recon data
		input_trt_latent = Zsample[idx_trt_type]

		### train optimal w-2 translation model
		model = SCVI_OptimTransPy36(196, model_c, device)
		for name, params in model.named_parameters():
			if name != 'condition_new':
				params.requires_grad = False
		opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
							   lr=0.001)

		try:
			losses = train_cinnTrans_model_gpu_py36(model, input_trt_latent_base, trt_onehot_base,
											   input_trt_latent, opt, device, iteration = 600)

			newfig = pd.DataFrame(losses, columns=['loss']).plot(figsize=[12, 9], fontsize=20).get_figure()
			plt.title("Loss: " + trt_type_base + "_to_obs_" + trt_type)
			newfig.savefig(os.path.join(path_save, "indice_" + str(0) + "_to_obs_" + str(indice_trt) + ".png"))

			res = model.condition_new.detach().numpy()
			np.save(os.path.join(path_save, "indice_" + str(0) + "_to_obs_" + str(indice_trt) + ".npy"), res)

			# w2 distance of the fitted value
			trt_onehot_otherTo = np.tile(res, (input_trt_latent_base.shape[0], 1))
			fake_latent_other, _ = perturbnet_model.trans_data(input_trt_latent_base, trt_onehot_base,
															   trt_onehot_otherTo)
			w2_val = w_dis_cal.calculate_fid_score(fake_latent_other, input_trt_latent)[0]
			pd_res_fittedW2Dis = pd.concat(
				[pd_res_fittedW2Dis, pd.DataFrame([[0, indice_trt, w2_val]],
												 columns=['start', 'target', 'W2'])])

			pd_res_fittedW2Dis.to_csv(os.path.join(path_save, 'pd_res_fittedW2Dis.csv'))

		except:
			pass

