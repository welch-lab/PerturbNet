API
=============

Cellular Representation Networks
---------------------------------
.. autosummary::
   :toctree: _autosummary
   :recursive:

   perturbnet.data_vae.VAE

Perturbation Representation Networks
-------------------------------------
.. autosummary::
   :toctree: _autosummary
   :recursive:

   perturbnet.chemicalvae.chemicalVAE.ChemicalVAE
   perturbnet.genotypevae.genotypeVAE.GenotypeVAE

cINNs
-------
.. autosummary::
   :toctree: _autosummary
   :recursive:
   perturbnet.cinn.flow.Net2NetFlow_TFVAEFlow
   perturbnet.cinn.flow.Net2NetFlow_TFVAE_Covariate_Flow
   perturbnet.cinn.flow.Net2NetFlow_scVIGenoFlow
   perturbnet.cinn.flow.Net2NetFlow_scVIFixFlow

Final Generative Models
---------------------------------
.. autosummary::
   :toctree: _autosummary
   :recursive:
   perturbnet.cinn.flow_generate.SCVIZ_CheckNet2Net
   perturbnet.cinn.flow_generate.TFVAEZ_CheckNet2Net

Feature Attribution
--------------------
.. autosummary::
   :toctree: _autosummary
   :recursive:
   
   perturbnet.cinn.FeatureAttr

Tools & Plot
-------------
.. autosummary::
   :toctree: _autosummary
   :recursive:

perturbnet.util.create_train_test_splits_by_key
perturbnet.util.prepare_embeddings_cinn
perturbnet.util.smiles_to_hot
perturbnet.util.contourplot_space_mapping
perturbnet.util.Seq_to_Embed_ESM

