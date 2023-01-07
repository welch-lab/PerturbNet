# PerturbNet

PerturbNet is a deep generative model that can predict the distribution of cell states induced by chemical or genetic perturbation. The repository contains the code for the preprint [PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations](https://www.biorxiv.org/content/10.1101/2022.07.20.500854v2). 

## System Requirements and Installation
PerturbNet works on Linux, Mac, or Windows. The key system requirements are Python (>3.7) and PyTorch (>1.7). TensorFlow is required for some functionality.
To install the package, simply install PyTorch (and TensorFlow if needed), then clone the repository. Expected installation time is about 10 minutes.

Some related module versions are: 
```
(1) Python: python3.8-anaconda/2020.07
(2) numpy: 1.18.5
(3) pandas 1.0.5
(4) scanpy: 1.8.1
(5) tensorflow: 1.14.0 
(6) matplotlib: 3.2.2
(7) scvi-tools: 0.7.1
(8) torch: 1.10.0
(9) umap-learn: 0.4.6
```

## Repository Structure and Usage

[`./net2net`](https://github.com/welch-lab/PerturbNet/tree/main/net2net) contains the conditional invertible neural network (cINN) modules in the [GitHub](https://github.com/CompVis/net2net/tree/master/net2net) repository of [Network-to-Network Translation with Conditional Invertible Neural Networks](https://arxiv.org/abs/2005.13580). 

[`./perturbnet`](https://github.com/welch-lab/PerturbNet/tree/main/perturbnet) contains the code to train the PerturbNet framework. We provide illustrations and guidance of how to use our repository for PerturbNet

[`./pytorch_scvi`](https://github.com/welch-lab/PerturbNet/tree/main/pytorch_scvi) contains our adapted modules to decode latent representations to expression profiles based on scVI version 0.7.1.

## Demo and Instructions for Usage
We have provided an example dataset on Dropbox and a Jupyter notebook showing how to run PerturbNet on the example dataset in 
[`./examples`](https://github.com/welch-lab/PerturbNet/tree/main/examples). 

## Reference

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
