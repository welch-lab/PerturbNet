# PerturbNet
PerturbNet is a deep generative model that can predict the distribution of cell states induced by chemical or genetic perturbation. The repository contains the code for the manuscript of [PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations](https://www.biorxiv.org/content/10.1101/2022.07.20.500854v2). 

# Structures and Usage

[`./net2net`] contains the conditional invertible neural network (cINN) modules in the [GitHub](https://github.com/CompVis/net2net/tree/master/net2net) repository of [Network-to-Network Translation with Conditional Invertible Neural Networks](https://arxiv.org/abs/2005.13580). 

[`./perturbnet`](https://github.com/welch-lab/PerturbNet/tree/main/perturbnet) contains the code to train the PerturbNet framework. We provide illustrations and guidance of how to user our repository for PerturbNet

[`./pytorch_scvi`](https://github.com/welch-lab/PerturbNet/tree/main/pytorch_scvi) contains our adapted modules to decode latent representations to expresssion profiles based on scVI version 0.7.1.


# Reference

Please consider citing

```
@article {Yu2022.07.20.500854,
	author = {Yu, Hengshi and Welch, Joshua D},
	title = {PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations},
	elocation-id = {2022.07.20.500854},
	year = {2022},
	doi = {10.1101/2022.07.20.500854},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/07/22/2022.07.20.500854},
	eprint = {https://www.biorxiv.org/content/early/2022/07/22/2022.07.20.500854.full.pdf},
	journal = {bioRxiv}
}

```
We appreciate your interest in our work. 
