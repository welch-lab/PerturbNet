# cINN modules and PerturbNet functions

- `module` has the modules of cINN flows (`flow.py`), prediction based on PerturbNet (`flow_generate.py`), optimal translation modules based on PerturbNet (`flow_optim_trans.py`). The `smiview_fun.py` is the adapted code from the [smiview package](https://pypi.org/project/smiview/) to find atom indices within a SMILES string. 

# PerturbNet Examples

We provide several training/evaluation examples for PerturbNet:

1. `chemvae_scvi_flow_train.py` and `chemvae_scvi_flow_eval.py` are training and evaluation files of PerturbNet for sci-Plex count cellular responses to chemical perturbations. 

2. `chemvae_vae_flow_train.py` and `chemvae_vae_flow_eval.py` are training and evaluation files of PerturbNet for LINCS-Drug normalized cellular responses to chemical perturbations. 

3. `genotypevae_scviBatchCorrected_flow_train.py` and `genotypevae_scviBatchCorrected_flow_eval.py` are training and evaluation files of PerturbNet with batch-correction steps for GSPS count cellular responses to genetic perturbations. 

4. `esm_scvi_flow_train.py` and `esm_scvi_flow_eval.py` are training and evaluation files of PerturbNet with fixed standard deviations of perturbation representations for Ursu count cellular responses to coding variants. 

We also provid application examples of PerturbNet:

1. `chemvae_vae_flow_conti_optimal_trans_train.py` and `chemvae_vae_flow_conti_optimal_trans_eval.py` have continuous optimal translation training and evaluation based on PerturbNet for LINCS-Drug data. 

2. `chemvae_vae_flow_disc_optimal_trans.py` has discrete optimal translation experiments based on PerturbNet for LINCS-Drug data. 

3. `chemvae_vae_flow_drug_score.py` computes attributions of chemical perturbations using integrated gradients based on PerturbNet for LINCS-Drug data. 


