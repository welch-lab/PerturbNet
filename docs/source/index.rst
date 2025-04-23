|GitHubStars| |PyPI| |PyPIDownloads|

PerturbNet
========================

PerturbNet is a deep generative framework designed to model and predict shifts in cell state—defined as changes in overall gene expression—in response to diverse cellular perturbations. PerturbNet consists of three trainable components: a perturbation representation network, a cellular representation network, and a conditional normalizing flow. These components work together to embed perturbations and cell states into a latent spaces and to learn a flexible mapping from perturbation features to gene expression distributions.

Given a perturbation of interest—such as gene knockdown, gene overexpression, sequence mutation, or drug treatment—PerturbNet predicts the resulting distribution of single-cell gene expression states. Currently, you can refer to the preprint `PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations <https://www.biorxiv.org/content/10.1101/2022.07.20.500854v2>`_. We will submit an updated version of the paper soon.

.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   usage
   api

.. toctree::
   :caption: Turtorial
   :maxdepth: 1
   :hidden:

   chemical_perturbation
   genetic_perturbation
   coding_variant
   feature_attribution
   GATA1_example
   



.. |GitHubStars| image:: https://img.shields.io/github/stars/welch-lab/PerturbNet?logo=GitHub&color=yellow
   :target: https://github.com/welch-lab/PerturbNet/stargazers

.. |PyPI| image:: https://img.shields.io/pypi/v/perturbnet?logo=PyPI
   :target: https://pypi.org/project/perturbnet/

.. |PyPIDownloads| image:: https://pepy.tech/badge/perturbnet
   :target: https://pepy.tech/project/perturbnet