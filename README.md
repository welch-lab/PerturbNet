# PerturbNet

PerturbNet is a deep generative model that can predict the distribution of cell states induced by chemical or genetic perturbation. Currently, you can refer to the preprint [PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations](https://www.biorxiv.org/content/10.1101/2022.07.20.500854v2). We will submit an updated version of the paper soon.  




## System Requirements and Installation

The current version of PerturbNet requires Python 3.7. All required dependencies are listed in ```requirements.txt```. We recommend creating a clean Conda environment using the following command:

```
conda create -n "PerturbNet" python=3.7
```
After setting up the environment, you can install the package by running:  
```
conda activate PerturbNet
pip install PerturbNet
```

##  Core Repository Structure

[`./perturbnet`](https://github.com/welch-lab/PerturbNet/tree/main/perturbnet) contains the core modules to train and benchmark the PerturbNet framework. 

[`./perturbnet/net2net`](https://github.com/welch-lab/PerturbNet/tree/main/net2net) contains the conditional invertible neural network (cINN) modules in the [GitHub](https://github.com/CompVis/net2net/tree/master/net2net) repository of [Network-to-Network Translation with Conditional Invertible Neural Networks](https://arxiv.org/abs/2005.13580). 


[`./perturbnet/pytorch_scvi`](https://github.com/welch-lab/PerturbNet/tree/main/pytorch_scvi) contains our adapted modules to decode latent representations to expression profiles based on scVI version 0.7.1.


## Tutorial and Reproducibility
The [`./notebooks`] directory contains Jupyter notebooks demonstrating how to use **PerturbNet** and includes code to reproduce the results:  
[Tutorial on using PerturbNet on chemical perturbations](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Tutorial_PerturbNet_Chemicals.ipynb)  
[Tutorial on using PerturbNet on genetic perturbations](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Tutorial_PerturbNet_Genetic.ipynb)  
[Tutorial on using PerturbNet on coding variants](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Tutorial_PerturbNet_coding_variants.ipynb)  
[Tutorial on using integrated gradients to calculate feature scores for chemicals](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Integrated_gradients_example.ipynb)  
[Benchmark on LINCS-Drug](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Benchmark_LINCS_Example.ipynb)  
[Benchmark on sci-Plex](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Benchmark_Sciplex_Example.ipynb)  
[Benchmark on Norman et al.](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Benchmark_Norman_Example.ipynb)  
[Benchmark on Ursu et al.](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Benchmark_Ursu_Example.ipynb)  
[Benchmark on Jorge et al.](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/Benchmark_Jorge_Example.ipynb)  
[Analysis of predicted novel GATA1 mutations](https://github.com/welch-lab/PerturbNet/blob/main/notebooks/GATA1_prediction_analysis.ipynb)  

The required data, toy examples, and model weights can be downloaded from [Hugging Face](https://huggingface.co/cyclopeta/PerturbNet_reproduce/tree/main).



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
