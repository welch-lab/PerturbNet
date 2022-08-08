#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import anndata as ad
import scvi

if __name__ == "__main__":
	path_data = ""
	adata = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))


	# train the model
	#scvi.data.setup_anndata(adata, layer = "counts", categorical_covariate_keys = ['gem_group'])
	scvi.data.setup_anndata(adata, layer = "counts", batch_key = 'gem_group')
	model = scvi.model.SCVI(adata, n_latent = 10)
	model.train(n_epochs = 400)
	model.save("./model/")

