# List of PerturbNet Files

- `util.py` contains the code for various generation metrics 

- `chemicalvae` contains code for ChemicalVAE 

- `cinn` contains code for cINN 

- `data_vae` contains code for cellular VAE models

- `genotypevae` contains code for GenotypeVAE

- `knn` contains code for KNN model 

# Usage

We train PerturbNet in three phases:

1. Train perturbation VAE models (`chemicalvae` or `genotypevae`)
2. Train cellular VAE (`data_vae`)
3. Train cINN model (`cinn`)


We evaluate PerturbNet prediction performance in examples in `cinn`. We also provide a KNN model (`knn`) with evaluation examples. 