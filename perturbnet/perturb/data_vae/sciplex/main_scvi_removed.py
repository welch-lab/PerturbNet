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

	perturb_with_onehot = list(input_ltpm_label['treatment'])
	removed_all_pers = np.load(os.path.join(path_wass, "RemovedPerturbs.npy"))

	kept_indices = [i for i in range(len(perturb_with_onehot)) if perturb_with_onehot[i] not in removed_all_pers]
	input_ltpm_label1 = input_ltpm_label.iloc[kept_indices, :]
	input_ltpm_label1.index = list(range(input_ltpm_label1.shape[0]))

	perturb_with_onehot = np.array(perturb_with_onehot)[kept_indices]

	sci_data_train = sci_data[kept_indices, :]
	sci_data = sci_data_train.copy()

	scvi.data.setup_anndata(sci_data, layer = "counts")

	model = scvi.model.SCVI(sci_data, n_latent=10)
	model.train(n_epochs=700)

	model.save("./model/")

 