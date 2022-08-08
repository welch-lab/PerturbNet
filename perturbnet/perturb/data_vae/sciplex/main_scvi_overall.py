# import the modules
import os
from scipy import sparse
import anndata as ad
from anndata import AnnData
import scvi
import pandas as pd
import numpy as np

# import the data, var, and obs to convert to anndata

if __name__ == "__main__":
	path_data = ""
	path_wass = ""

	sci_data = ad.read_h5ad(os.path.join(path_data, "data.h5ad"))

	input_ltpm_label = sci_data.obs

	scvi.data.setup_anndata(sci_data, layer = "counts")

	model = scvi.model.SCVI(sci_data, n_latent=10)
	model.train(n_epochs=700)

	model.save("./model/")

 