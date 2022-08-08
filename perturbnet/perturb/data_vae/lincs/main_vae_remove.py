#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('..')

import random 
import numpy as np
import pandas as pd

from scipy import sparse
from modules.vae import *

import umap
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":

	model_save_path = "model_vae_removed"
	path_save = "output_vae_removed"
	if not os.path.exists(path_save):
		os.makedirs(path_save, exist_ok = True)

	path_data = ""
	path_onehot = ""

	usedata = np.load(os.path.join(path_data, "data.npy"))

	## with canonical smiles
	indices_with_value = np.load(os.path.join(path_onehot, "idx.npy"))
	usedata = usedata[indices_with_value]
	metadata = pd.read_csv(os.path.join(path_data, "PerturbMeta.csv"))
	metadata = metadata.iloc[indices_with_value, :]

	## with onehot
	data_onehot_perturb = np.load(os.path.join(path_onehot, "SmilesOneHot.npy"))
	data_smiles_perturb = np.load(os.path.join(path_onehot, "Smiles.npy"))

	list_canonSmiles = list(metadata["canonical_smiles"])
	data_smiles_perturb_set = set(data_smiles_perturb)
	indicesWithOnehot = [i for i in range(len(list_canonSmiles)) if list_canonSmiles[i] in data_smiles_perturb_set]

	usedata = usedata[indicesWithOnehot]
	metadata = metadata.iloc[indicesWithOnehot, :]

	# remove 2,000 perturbations 
	perturb_with_onehot = list(metadata["canonical_smiles"])
	removed_all_pers = np.load(os.path.join(path_onehot, "RemovedPerturbs.npy"))
	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	removed_indices = np.setdiff1d(list(range(len(perturb_with_onehot))), kept_indices)
		#[i for i in range(len(perturb_with_onehot)) if i not in set(kept_indices)]

	usedata_removed = usedata[removed_indices]
	usedata  = usedata[kept_indices]
	metadata = metadata.iloc[kept_indices, :]
	perturb_with_onehot = np.array(perturb_with_onehot)[kept_indices]

	## VAE training part
	vae = VAE(num_cells_train = usedata.shape[0], 
			  x_dimension = usedata.shape[1], 
			  learning_rate = 1e-4, 
			  BNTrainingMode = False)
	
	vae.train_np(train_data = usedata,
				 use_test_during_train = True, 
				 test_every_n_epochs = 50, 
				 n_epochs = 201, 
				 save = True, 
				 model_save_path = model_save_path, 
				 output_save_path = path_save, 
				 verbose = False)

	# Evaluation
	gen_visual = MetricVisualize()

	## generation umap
	test_size = 3000
	sampled_indices = random.sample(range(len(usedata)), test_size)
	test_data = usedata[sampled_indices]
	gen_data = vae.reconstruct(test_data)

	path_file_save = os.path.join(path_save, 'vae_gen.png')
	gen_visual.umapPlot(test_data, gen_data, path_file_save)

	## removed
	sampled_indices = random.sample(range(len(removed_indices)), test_size)
	test_data = usedata_removed[sampled_indices]
	gen_data = vae.reconstruct(test_data)

	path_file_save = os.path.join(path_save, 'vae_gen_removed.png')
	gen_visual.umapPlot(test_data, gen_data, path_file_save)


	## training curves
	train_loss = np.array(vae.train_loss)
	valid_loss = np.array(vae.valid_loss)
	train_time = round(vae.training_time/60.0, 1)

	trainvalid_loss = np.concatenate([train_loss.reshape([train_loss.shape[0], 1]), 
									  valid_loss.reshape([valid_loss.shape[0], 1])], axis = 1)

	trainvalid_loss_pd = pd.DataFrame(trainvalid_loss)
	trainvalid_loss_pd.columns = ['train_loss', 'valid_loss']
	trainvalid_loss_pd['Epoch'] = list(range(trainvalid_loss_pd.shape[0]))

	newfig = trainvalid_loss_pd.set_index('Epoch').plot(figsize = [8,6 ], fontsize = 20).get_figure()
	plt.title("Minutes: " + str(train_time))
	newfig.savefig(os.path.join(path_save, 'losses.png'))