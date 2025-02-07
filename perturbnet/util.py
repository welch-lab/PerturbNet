import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import scvi
import scanpy as sc
import time

from anndata import AnnData
from scipy import linalg, stats, sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, r2_score
from sklearn.preprocessing import label_binarize
from math import sqrt
from scipy.stats import gaussian_kde

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, QED
from rdkit import DataStructs
import argparse
import pickle
from tqdm import tqdm

sys.path.append('../')
from perturbnet.util import * 
from perturbnet.cinn.flow import * 
from perturbnet.genotypevae.genotypeVAE import *
from perturbnet.data_vae.vae import *
from perturbnet.cinn.flow_generate import SCVIZ_CheckNet2Net
matplotlib.use('agg')










class NormalizedRevisionRSquare:
    """
    A class to calculate the R-squared and Pearson correlation metrics for normalized count data.

    This class provides methods for comparing real and synthetic count data through various 
    approaches, using both R-squared and Pearson correlation as evaluation metrics. 
    Data normalization is performed by row, followed by log transformation.

    Parameters
    ----------
    largeCountData : numpy.ndarray
        A 2D array of count data used to initialize the column-wise normalization 
        parameters.
    targetSize : float, optional
        A scaling factor for row-normalized data (default is 1e4). This factor is applied
        to the normalized data before the log transformation.

    Attributes
    ----------
    targetSize : float
        Scaling factor for normalized data.
    col_mu : numpy.ndarray
        The mean of each column in the normalized `largeCountData`, used for column-wise 
        normalization. The column normalization has been removed
    col_std : numpy.ndarray
        The standard deviation of each column in the normalized `largeCountData`, used for 
        column-wise normalization. The column normalization has been removed.

    Methods
    -------
    calculate_largeData(largeCountData)
        Initializes the `col_mu` and `col_std` by normalizing `largeCountData` by row sums, 
        scaling by `targetSize`, and applying a log transformation.
    
    calculate_pearson(real_data, fake_data)
        Calculates the Pearson correlation coefficient between the average normalized 
        real and fake data. Returns 1.5 if NaN values are encountered.

    calculate_r_square(real_data, fake_data)
        Calculates the R-squared metric between the average normalized real and fake data.
        Returns the R-squared value, the normalized real data, and the normalized predicted data.
    
    calculate_r_square_real_total(real_data, fake_data)
        Calculates the R-squared metric using real data as the true values for comparison.
        Returns the R-squared value, normalized real data, and normalized predicted data.
    
    calculate_r_square_norm_by_all(real_data, fake_data, norm_vec_real, norm_vec_fake)
        Calculates the R-squared metric by normalizing real and predicted data using external
        normalization vectors. Returns the R-squared value. Applied when the all genes are
        available.

    calculate_pearson_norm_by_all(real_data, fake_data, norm_vec_real, norm_vec_fake)
        Calculates the Pearson correlation coefficient by normalizing real and predicted data
        using external normalization vectors. Returns the Pearson correlation coefficient.
        Applied when the all genes are available.
    """

    def __init__(self, largeCountData, targetSize = 1e4):
        
        self.targetSize = targetSize
        self.calculate_largeData(largeCountData)
        
    def calculate_largeData(self, largeCountData):
        usedata = largeCountData.copy()
        usedata = usedata / usedata.sum(axis = 1)[:, None] * self.targetSize
        usedata = np.log1p(usedata)

        self.col_mu = usedata.mean(axis = 0)
        self.col_std = usedata.std(axis = 0)

    
    def calculate_pearson(self, real_data, fake_data):
        real_data_norm = real_data.copy()
        real_data_norm_sum = real_data_norm.sum(axis = 1) 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]
        
        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = fake_data_norm.sum(axis = 1) 
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]
        
        real_data_norm = real_data_norm / real_data_norm.sum(axis = 1)[:, None] * self.targetSize
        fake_data_norm = fake_data_norm / fake_data_norm.sum(axis = 1)[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
        x = np.average(real_data_norm, axis = 0)
        y = np.average(fake_data_norm, axis = 0)
        if (np.isnan(x).any() or np.isnan(y).any()):
            return 1.5
        else:
            m, b, r_value, p_value, std_err = stats.linregress(x, y)

            return r_value



    def calculate_r_square(self, real_data, fake_data):
        real_data_norm = real_data.copy()
        real_data_norm_sum = real_data_norm.sum(axis = 1) 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]
        
        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = fake_data_norm.sum(axis = 1) 
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]
        
        real_data_norm = real_data_norm / real_data_norm.sum(axis = 1)[:, None] * self.targetSize
        fake_data_norm = fake_data_norm / fake_data_norm.sum(axis = 1)[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
        x = np.average(real_data_norm, axis = 0)
        y = np.average(fake_data_norm, axis = 0)
        if (np.isnan(x).any() or np.isnan(y).any()):
            return 1.5, 1.5 , 1.5
        else:            
            r2_value = r2_score(x, y)

            return r2_value, real_data_norm , fake_data_norm  
    
    def calculate_r_square_real_total(self, real_data, fake_data):
        real_data_norm = real_data.copy()
        real_data_norm_sum = real_data_norm.sum(axis = 1) 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]
        
        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = fake_data_norm.sum(axis = 1) 
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]
        
        real_data_norm = real_data_norm / real_data_norm.sum(axis = 1)[:, None] * self.targetSize
        fake_data_norm = fake_data_norm / fake_data_norm.sum(axis = 1)[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
        # important to make sure x is y_true and y is y_pred since it will affect which y_bar to be used
        x = np.average(real_data_norm, axis = 0)
        y = np.average(fake_data_norm, axis = 0)
        r2_value = r2_score(x, y)

        return r2_value, real_data_norm , fake_data_norm 
    
    def calculate_r_square_norm_by_all(self, real_data, fake_data, norm_vec_real, norm_vec_fake):
        real_data_norm = real_data.copy()
        real_data_norm_sum = norm_vec_real 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = norm_vec_fake
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]

        real_data_norm = real_data_norm /  real_data_norm_sum[:, None] * self.targetSize
        fake_data_norm = fake_data_norm /  fake_data_norm_sum[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
        x = np.average(real_data_norm, axis = 0)
        y = np.average(fake_data_norm, axis = 0)
        r2_value = r2_score(x, y)
        return(r2_value)
    
    def calculate_pearson_norm_by_all(self, real_data, fake_data,norm_vec_real,norm_vec_fake):
        real_data_norm = real_data.copy()
        real_data_norm_sum = norm_vec_real 
        real_data_norm  = real_data_norm[real_data_norm_sum != 0,:]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = norm_vec_fake
        fake_data_norm  = fake_data_norm[fake_data_norm_sum != 0,:]

        real_data_norm = real_data_norm /  real_data_norm_sum[:, None] * self.targetSize
        fake_data_norm = fake_data_norm /  fake_data_norm_sum[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
        x = np.average(real_data_norm, axis = 0)
        y = np.average(fake_data_norm, axis = 0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        return(r_value)

    





class fidscore:
    """
    A class to calculate the Frechet Inception Distance (FID) score and additional
    metrics for comparing distributions of real and predicted data representations. 

    This class provides methods to compute the FID score, which measures the similarity 
    between two distributions (real and generated data) based on their statistics 
    in a lower-dimensional space (e.g., after PCA transformation). However, because 
    our adjusted FID has limitation to compare models using different latent space,we don't
    use this metirc anymore。
    
    Instead, R-squared and Pearson correlation metrics can be computed for a more detailed 
    comparison.

    Attributes
    ----------
    pca_50 : sklearn.decomposition.PCA
        PCA transformer set to reduce data dimensionality to 50 components, with a 
        random seed for reproducibility.

    Methods
    -------
    calculate_statistics(numpy_data)
        Calculates the mean and covariance matrix of the given data.
    
    calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
        Computes the Frechet Distance between two distributions specified by their 
        mean and covariance matrices. Returns both the FID score and an error flag 
        for complex matrix values.

    calculate_fid_score(real_data, fake_data, pca_data_fit=None, if_dataPC=False)
        Computes the FID score between the distributions of real and fake data 
        by applying PCA transformation and calculating the Frechet Distance in 
        the reduced space.

    calculate_r_square(real_data, fake_data)
        Calculates the R-squared metric between the average values of real and 
        predicted data. Applied for normalized data.

    calculate_pearson(real_data, fake_data)
        Calculates the Pearson correlation coefficient between the average values 
        of real and predicted data distributions. Applied for normalized data.

    """

    def __init__(self):
        super().__init__()
        self.pca_50 = PCA(n_components=50, random_state = 42)

    def calculate_statistics(self, numpy_data):
        mu = np.mean(numpy_data, axis = 0)
        sigma = np.cov(numpy_data, rowvar = False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps = 1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp = False)
        if not np.isfinite(covmean).all():
            msg = (
                'fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates' % eps
            )
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol = 1e-3):
                m = np.max(np.abs(covmean.imag))
                #raise ValueError('Cell component {}'.format(m))
            covmean = covmean.real

            image_error = 1
        else:
            image_error = 0
        
        tr_covmean = np.trace(covmean)


        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean ), image_error


    def calculate_fid_score(self, real_data, fake_data, pca_data_fit = None, if_dataPC = False):

        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis = 0)

        if if_dataPC:
            pca_all = pca_data_fit.transform(all_data)
        else:
            pca_all = self.pca_50.fit(real_data).transform(all_data)
        data1, data2 = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)
        
        return fid_value, image_error

    def calculate_r_square(self, real_data, fake_data):
        real_data_sum = real_data.sum(axis = 1) 
        real_data = real_data[real_data_sum != 0,:]
        fake_data_sum = fake_data.sum(axis = 1) 
        fake_data = fake_data[fake_data_sum != 0,:]
        x = np.average(real_data, axis = 0)
        y = np.average(fake_data, axis = 0)
        r2_value = r2_score(x, y)


        return r2_value


    def calculate_pearson(self, real_data, fake_data):
        real_data_sum = real_data.sum(axis = 1) 
        real_data = real_data[real_data_sum != 0,:]
        fake_data_sum = fake_data.sum(axis = 1) 
        fake_data = fake_data[fake_data_sum != 0,:]
        x = np.average(fake_data, axis = 0)
        y = np.average(real_data, axis = 0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)

        return r_value 


class fidscore_scvi_extend(fidscore):
    """
    calculate FID score metric defined on the scVI latent space
    """

    def __init__(self, scvi_model):
        super().__init__()

        self.scvi_model = scvi_model

    def calculate_fid_scvi_score(self, real_data, fake_data, give_mean = False):

        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1
            
        all_data = np.concatenate([fake_data, real_data], axis = 0)
        all_adata = AnnData(X = all_data)
        all_adata.layers["counts"] = all_adata.X.copy()
        
        repre_all = self.scvi_model.get_latent_representation(adata = all_adata, give_mean = give_mean)
        
        data1, data2 = repre_all[fake_data.shape[0]:], repre_all[:fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error  = self.calculate_frechet_distance(m1, s1, m2, s2)
        
        return fid_value, image_error


    def calculate_fid_scvi_score_with_y(self, real_data, fake_data, y_data, y_key, scvi, adata, give_mean = False):

        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis = 0)
        all_adata = AnnData(X = all_data)
        all_adata.layers["counts"] = all_adata.X.copy()
        all_adata.obs[y_key] = np.concatenate((y_data, y_data))
        #scvi.data.setup_anndata(all_adata, layer = "counts", batch_key = y_key)
        scvi.data.transfer_anndata_setup(adata, all_adata)

        repre_all = self.scvi_model.get_latent_representation(adata = all_adata, give_mean = give_mean)

        data1, data2 = repre_all[fake_data.shape[0]:], repre_all[:fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error  = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error


    def calculate_fid_scvi_indice_score(self, real_indices, fake_indices, give_mean = False):

        if len(real_indices) <= 1 or len(fake_indices) <= 1:
            return -1, 1

        n_fake = len(fake_indices)
        all_indices = np.concatenate([np.array(fake_indices), np.array(real_indices)])
        repre_all = self.scvi_model.get_latent_representation(indices = all_indices, give_mean = give_mean)

        data1, data2 = repre_all[n_fake:], repre_all[:n_fake]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error  = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error


class fidscore_scgen_extend(fidscore):
    """
    calculate FID score metric defined on the scGen latent space
    """

    def __init__(self, scgen_model):
        super().__init__()
        self.scgen_model = scgen_model

    def calculate_fid_scgen_score(self, real_data, fake_data, if_count_layer = False, give_mean=False):
        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis=0)
        all_adata = AnnData(X = all_data)

        if if_count_layer:
            all_adata.layers["counts"] = all_adata.X.copy()

        repre_all = self.scgen_model.get_latent_representation(adata=all_adata, give_mean=give_mean)

        data1, data2 = repre_all[fake_data.shape[0]:], repre_all[:fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error

class fidscore_vae_extend(fidscore):
    """
    calculate FID score metric defined on the regular VAE latent space
    """

    def __init__(self, sess, z_gen_data_v, z_gen_mean_v, X_v, is_training):
        super().__init__()

        self.sess = sess
        self.z_gen_data_v = z_gen_data_v
        self.X_v = X_v
        self.z_gen_mean_v = z_gen_mean_v
        self.is_training = is_training

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps = 1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp = False)
        if not np.isfinite(covmean).all():
            msg = (
                'fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates' % eps
            )
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol = 1e-3):
                m = np.max(np.abs(covmean.imag))
                
                #raise ValueError('Cell component {}'.format(m))
            covmean = covmean.real

            image_error = 1
        else:
            image_error = 0


        tr_covmean = np.trace(covmean)


        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean ), image_error


    def calculate_fid_vae_score(self, real_data, fake_data, give_mean = False):
        
        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis = 0)
        # all_adata = AnnData(X = all_data)
        # all_adata.layers["counts"] = all_adata.X.copy()
        
        # repre_all = self.scvi_model.get_latent_representation(adata = all_adata, give_mean = give_mean)
        feed_dict = {self.X_v: all_data, self.is_training: False}
        
        if give_mean:
            repre_all = self.sess.run(self.z_gen_mean_v, feed_dict = feed_dict)
        else:
            repre_all = self.sess.run(self.z_gen_data_v, feed_dict = feed_dict)

        data1, data2 = repre_all[fake_data.shape[0]:], repre_all[:fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)
        
        return fid_value, image_error







class Standardize:
    """
    standardize latent space
    """

    def __init__(self, data_all, model, device):

        self.onehot_data = self.random_select(data_all)
        self.model = model 
        self.device = device
        self.estimate_stand()

    def estimate_stand(self):
        batch_size = 2500

        Z_sample = np.zeros((len(self.onehot_data), 196))
        
        for chunk in self.chunks(list(range(len(self.onehot_data))), batch_size):
            one_hot = torch.tensor(self.onehot_data[chunk]).float().to(self.device)
            _, _, _, embdata_torch = self.model(one_hot)
            Z_sample[chunk, :] = embdata_torch.cpu().detach().numpy()
        self.mu = np.mean(Z_sample, axis = 0)
        self.std = np.std(Z_sample, axis = 0)
        return

    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def standardize_z_torch(self, z):
        return (z - torch.tensor(self.mu).float().to(self.device)) / torch.tensor(self.std).float().to(self.device)

    def random_select(self, data, size = 50000):
        indices = random.sample(range(len(data)), size)
        return data[indices]

    @staticmethod
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i: (i + n)]



class StandardizeLoad:
    """
    standardize latent space with known global means and stds
    """
    def __init__(self, mu, std, device):
        self.mu = mu
        self.std = std
        self.device = device

    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def standardize_z_torch(self, z):
        return (z - torch.tensor(self.mu).float().to(self.device)) / torch.tensor(self.std).float().to(self.device)


class ConcatDatasetWithIndices(torch.utils.data.Dataset):
    """
    data structure with sample indices of two datasets
    """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple([d[i] for d in self.datasets] + [i])

    def __len__(self):
        return min(len(d) for d in self.datasets)


####################################new functions###################

def smiles_to_hot(smiles, max_len, padding, nchars,char_list = None):
    """
    Converts SMILES strings to a one-hot encoded 3D numpy array representation.

    This function takes a list of SMILES (Simplified Molecular Input Line Entry System)
    strings and converts each character in the SMILES sequence to a one-hot encoded 
    vector. The function allows specification of the character set used, the maximum 
    sequence length, and the type of padding to ensure consistent length among 
    sequences.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings to be converted.
    max_len : int
        The maximum length of each SMILES string. Shorter strings will be padded, 
        while longer strings will cause errors.
    padding : str
        Specifies the padding strategy. Options are 'left' or 'right', determining 
        whether padding is added to the beginning or end of each SMILES string.
    nchars : int
        The number of unique characters in the SMILES alphabet, defining the 
        third dimension of the output array.
    char_list : list of str, optional
        A list of unique characters expected in the SMILES strings. If not provided, 
        a default character list will be used.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array of shape `(num_smiles, max_len, nchars)`, where each 
        entry is a one-hot encoded representation of a SMILES character.
    """
    
    
    if char_list == None:
        char_list = ["7", "6", "o", "]", "3", "s", "(", "-", "S", "/", "B", "4", "[", ")", "#", "I", "l", "O", "H", "c", "1", "@", "=", "n", "P", "8", "C", "2", "F", "5", "r", "N", "+", "\\", " "]
    char_indices = {}
    for i in range(len(char_list)):
        char_indices[char_list[i]] = i
    
    smiles = [pad_smile(i, max_len, padding)
              for i in smiles if pad_smile(i, max_len, padding)]

    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)

    for i, smile in tqdm(enumerate(smiles)):
        for t, char in enumerate(smile):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", smile)
                raise e
    return X


def pad_smile(string, max_len, padding='right'):
    """
    Pads a SMILES string to a specified maximum length.

    This function pads a given SMILES string to a fixed length by adding 
    spaces either to the right or left, based on the specified padding 
    strategy. If no padding is needed (i.e., the string length equals or 
    exceeds `max_len`), the original string is returned.

    Parameters
    ----------
    string : str
        The SMILES string to be padded.
    max_len : int
        The maximum length of the output string. If the input string is 
        shorter than `max_len`, it will be padded with spaces.
    padding : str, optional
        The padding strategy to use, either 'right', 'left', or 'none' 
        (default is 'right'). If 'right', spaces are added to the end 
        of the string; if 'left', spaces are added to the beginning; 
        if 'none', the original string is returned without any padding.

    Returns
    -------
    str
        The padded SMILES string if `padding` is 'right' or 'left', or 
        the original string if `padding` is 'none' or if `string` length 
        equals or exceeds `max_len`.
    """
    
    if len(string) <= max_len:
        if padding == 'right':
            return string + " " * (max_len - len(string))
        elif padding == 'left':
            return " " * (max_len - len(string)) + string
        elif padding == 'none':
            return string
        




def Seq_to_Embed_ESM(ordered_trt, batch_size, model, alphabet, save_path = None):
    """
    Converts sequences to embeddings using the ESM (Evolutionary Scale Modeling) model.

    This function processes a list of sequences in batches, generating embeddings 
    for each sequence using a specified ESM model. The embeddings are calculated 
    as the mean of token representations for each sequence, excluding padding 
    tokens. Optionally, embeddings can be saved to a specified path.

    Parameters
    ----------
    ordered_trt : list of str
        List of sequences to be embedded in order.
    batch_size : int
        The number of sequences to process in each batch.
    model : torch.nn.Module
        The pre-trained ESM model used to generate embeddings.
    alphabet : ESMAlphabet
        Alphabet instance associated with the ESM model, used for tokenization and 
        batch conversion.
    save_path : str, optional
        Path to save the resulting embeddings as a `.npy` file. If not provided, 
        embeddings are not saved.

    Returns
    -------
    list of numpy.ndarray
        A list containing the embeddings for each sequence.

    """
        
        
    data = []
    count = 1
    batch_converter = alphabet.get_batch_converter()
    for i in ordered_trt:
        data.append((count,i))
        count += 1

    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    for j in tqdm(range(len(batches))):
        batch = batches[j]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len + 1].mean(0).numpy().reshape(1, -1))
    if save_path:
        np.save(save_path,sequence_representations)
    
    return(sequence_representations)


def create_train_test_splits_by_key(adata, train_ratio, add_key, split_key, control, random_seed=None):
    """
    Splits the observations in an AnnData object into training and testing sets based on unique values of a specified key,
    with certain control values always included in the training set.
    
    Parameters:
    adata (AnnData): The AnnData object containing the dataset.
    train_ratio (float): The proportion of unique values to include in the train split (0 < train_ratio < 1), excluding control values.
    add_key (str): The key to be added to adata.obs, where the train/test labels will be stored.
    split_key (str): The key in adata.obs used to determine the unique values for making splits.
    control (list): A list of values from the split_key that should always be included in the training set.
    random_seed (int, optional): The seed for the random number generator for reproducibility. If None, the seed is set based on the current time.

    Returns:
    None: The function adds a new column to adata.obs indicating train/test split.
    """
    if random_seed is None:
        random_seed = int(time.time())
    np.random.seed(random_seed)
    unique_values = adata.obs[split_key].unique()
    non_control_values = [value for value in unique_values if value not in control]
    np.random.shuffle(non_control_values)
    num_train = int(np.floor(train_ratio * len(non_control_values)))
    train_values = set(non_control_values[:num_train])
    train_values.update(control)
    adata.obs[add_key] = adata.obs[split_key].apply(lambda x: 'train' if x in train_values else 'test')
    
def prepare_embeddings_cinn(adata, perturbation_key, trt_key, embed_key):
    
    """
    Prepare perturbation embeddings for CINN (Cell Inference of Network Navigation) analysis.

    This function extracts perturbation labels, retrieves corresponding embeddings, and creates a mapping 
    between perturbation types and their corresponding embedding indices. It is typically used in the context 
    of perturbation experiments, where the goal is to relate perturbations to learned embeddings for further analysis.

    Parameters:
    - adata (AnnData): The AnnData object containing the dataset, where perturbation information is stored in `.obs` 
      and embeddings are stored in `.uns`.
    - perturbation_key (str): The key in `adata.obs` that stores the perturbation labels (e.g., gene knockdowns, drug treatments).
    - trt_key (str): The key in `adata.uns` where treatment or perturbation types are stored. This is typically an array 
      indicating different treatments corresponding to each observation.
    - embed_key (str): The key in `adata.uns` where the embeddings (e.g., from a dimensionality reduction method like PCA or UMAP) 
      corresponding to each treatment are stored.

    """
    
    perturb_with_onehot = np.array(adata.obs[perturbation_key])
    trt_list = np.unique(perturb_with_onehot)
    embed_idx = []
    for i in range(len(trt_list)):
        trt = trt_list[i]
        idx = np.where(adata.uns[trt_key] == trt)[0][0]
        embed_idx.append(idx)
    embeddings = adata.uns[embed_key][embed_idx]
        
    perturbToEmbed = {}
    for i in range(trt_list.shape[0]):
        perturbToEmbed[trt_list[i]] = i
        
    return perturb_with_onehot, embeddings, perturbToEmbed
    


    

#####################################
# PLOT FUNCTION
######################################
def umapPlot_latent_check(real_latent, fake_latent, path_file_save = None):
    """
    Creates a UMAP plot to visually compare real and fake latent embeddings. This function reduces the dimensionality
    of the concatenated real and fake latent vectors and plots them in two-dimensional space, with points colored by
    category.

    Parameters:
    - real_latent (np.array): Array of latent vectors representing real data.
    - fake_latent (np.array): Array of latent vectors representing fake data.
    - path_file_save (str, optional): File path to save the generated UMAP plot image. If None, the plot will not be
      saved. Defaults to None.

    Returns:
    - chart_pr (ggplot): A ggplot object representing the UMAP plot, with points colored by 'Real' or 'Fake' category.

    """
    
    all_latent = np.concatenate([fake_latent, real_latent], axis = 0)
    cat_t = ["Real"] * real_latent.shape[0]
    cat_g = ["Fake"] * fake_latent.shape[0]
    cat_rf_gt = np.append(cat_g, cat_t)
    trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(all_latent)
    X_embedded_pr = trans.transform(all_latent)
    df = X_embedded_pr.copy()
    df = pd.DataFrame(df)
    df['x-umap'] = X_embedded_pr[:,0]
    df['y-umap'] = X_embedded_pr[:,1]
    df['category'] = cat_rf_gt
    
    chart_pr = ggplot(df, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
    + geom_point(size=0.5, alpha = 0.5) \
    + ggtitle("UMAP dimensions")

    if path_file_save is not None:
        chart_pr.save(path_file_save, width=12, height=8, dpi=144)
    return chart_pr


def boxplot_metrics(model_dict, metric_key, path_file_save = None):
    """
    Generates a box plot to compare a specified metric across multiple models. The function takes a dictionary of
    model names and their corresponding DataFrames containing performance metrics, then plots the metric of interest
    for each model side-by-side in a box plot format.

    Parameters:
    - model_dict (dict): Dictionary where keys are model names and values are DataFrames containing model metrics. Each
      DataFrame should include columns for 'perturbation' and the specified `metric_key`.
    - metric_key (str): Name of the metric column to be used for the box plot comparison across models.
    - path_file_save (str, optional): File path to save the generated box plot image. If None, the plot will not be
      saved. Defaults to None.

    Returns:
    - chart_pr (ggplot): A ggplot object representing the box plot for the specified metric across models.

    """
    
    shared_cols = ["perturbation", metric_key]
    df_list = []
    for model, results in model_dict.items():
        results = results[shared_cols]
        results["model"] = np.repeat(model,results.shape[0])
        df_list.append(results)
    df = pd.concat(df_list, ignore_index=True)
    
    chart_pr = ggplot(df, aes(x= "model", y= metric_key, fill = "model") ) \
    + geom_boxplot() \

    if path_file_save is not None:
        chart_pr.save(path_file_save, width=12, height=8, dpi=144)
    return chart_pr





def contourplot_space_mapping(embeddings_cell, embeddings_pert, background_pert, background_cell, highlight_labels,
                              colors, random_state=42, bandwidth=0.2,save_path = None, dpi = 300, figsize = (12, 6),
                                       save_embed = False, save_path_embed = None, Y_embedded = None, Z_embedded = None):
    
    """
    Generates a contour plot mapping of embeddings for perturbation and cellular representations. The function uses
    UMAP for dimensionality reduction and visualizes both background and highlighted embeddings, optionally saving the
    embeddings and plot.

    Parameters:
    - embeddings_cell (np.array): Array of embeddings representing cell data to be plotted with highlights.
    - embeddings_pert (np.array): Array of embeddings representing perturbation data to be plotted with highlights.
    - background_pert (np.array): Background data for perturbation embeddings to provide additional context in the plot.
    - background_cell (np.array): Background data for cell embeddings to provide additional context in the plot.
    - highlight_labels (list): List of labels to highlight in the plot, differentiating these points from the background.
    - colors (list): List of colors corresponding to each highlight label for visual distinction.
    - random_state (int, optional): Seed for UMAP dimensionality reduction to ensure reproducibility. Defaults to 42.
    - bandwidth (float, optional): Bandwidth parameter for Gaussian KDE used in contouring highlighted points. Defaults to 0.2.
    - save_path (str, optional): Path to save the generated plot image. If None, the plot will not be saved. Defaults to None.
    - dpi (int, optional): Dots per inch for the saved plot image. Only used if save_path is specified. Defaults to 300.
    - figsize (tuple, optional): Size of the figure (width, height) in inches. Defaults to (12, 6).
    - save_embed (bool, optional): Whether to save the UMAP-transformed embeddings to files. Defaults to False.
    - save_path_embed (str, optional): Path to save UMAP-transformed embeddings if save_embed is True. Defaults to None.
    - Y_embedded (np.array, optional): Precomputed UMAP embeddings for perturbation data. If None, embeddings are computed.
    - Z_embedded (np.array, optional): Precomputed UMAP embeddings for cell data. If None, embeddings are computed.

    """        

    embeddings_cell_all = np.concatenate([background_cell,embeddings_cell ])
    embeddings_pert_all = np.concatenate([background_pert,embeddings_pert  ])

    cat_pert = ["Other"] * background_pert.shape[0] + [label for label in highlight_labels for _ in range(embeddings_pert.shape[0] // len(highlight_labels))]
    cat_cell = ["Other"] * background_cell.shape[0] + [label for label in highlight_labels for _ in range(embeddings_cell.shape[0] // len(highlight_labels))]

    # Create UMAP transformers and transform data
    if Y_embedded == None and Z_embedded == None:
        trans_pert = umap.UMAP(random_state=random_state, min_dist=0.5, n_neighbors=30).fit(embeddings_pert_all)
        trans_cell = umap.UMAP(random_state=random_state, min_dist=0.5, n_neighbors=30).fit(embeddings_cell_all)
        Y_embedded = trans_pert.transform(embeddings_pert_all)
        if save_embed:
            np.save(save_path_embed + "Y_embedded.npy",Y_embedded)
        Z_embedded = trans_cell.transform(embeddings_cell_all)
        if save_embed:
            np.save(save_path_embed + "Z_embedded.npy",Z_embedded)

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # Define a plotting function for each subplot
    def plot_with_contours(ax, data, categories, title, highlights, colors, add_contour=True):
        for highlight, color in zip(highlights, colors):
            highlight_data = data[categories == highlight]
            other_data = data[categories != highlight]

            # Plot background data
            ax.scatter(other_data[:, 0], other_data[:, 1], color='gray', s=1, label='Other')

            # Plot highlight data
            ax.scatter(highlight_data[:, 0], highlight_data[:, 1], color=color, s=1, label=highlight)

            if add_contour and highlight_data.size > 0:
                x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
                y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
                x_grid = np.linspace(x_min, x_max, 100)
                y_grid = np.linspace(y_min, y_max, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                kde = gaussian_kde(highlight_data.T, bw_method=bandwidth)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contour(X, Y, Z, levels=5, colors=color)

        ax.set_xlim(np.min(data[:, 0]), np.max(data[:, 0]))
        ax.set_ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plotting for each representation
    plot_with_contours(ax1, Y_embedded, np.array(cat_pert), 'Perturbation Representation', highlight_labels, colors, add_contour=False)
    plot_with_contours(ax2, Z_embedded, np.array(cat_cell), 'Cellular Representation', highlight_labels, colors)

    # Draw lines for highlighted points
    transFigure = fig.transFigure.inverted()
    highlight_indices = np.where(np.isin(np.array(cat_pert), highlight_labels))[0]
    
    if len(highlight_indices) > 30:
        highlight_indices = np.random.choice(highlight_indices, 30, replace=False)  # Randomly pick 30 indices
        
    pert_indices = highlight_indices 
    if background_cell.shape[0] > background_pert.shape[0]:
        Y_indices = highlight_indices - background_cell.shape[0] + background_pert.shape[0]
        Z_indices = highlight_indices
    
    elif background_cell.shape[0] < background_pert.shape[0]:
        Z_indices = highlight_indices - background_pert.shape[0] + background_cell.shape[0]
        Y_indices = highlight_indices
    else:
        Y_indices = highlight_indices
        Z_indices = highlight_indices
        
        
        
    for i in range(len(highlight_indices)):
        index = highlight_indices[i]
        xy1 = transFigure.transform(ax1.transData.transform(Y_embedded[Y_indices[i]]))
        xy2 = transFigure.transform(ax2.transData.transform(Z_embedded[Z_indices[i]]))
        line_color = colors[highlight_labels.index(cat_pert[pert_indices[i]])]
        line = matplotlib.lines.Line2D((xy1[0], xy2[0]), (xy1[1], xy2[1]), transform=fig.transFigure, color=line_color, linewidth=0.5)
        fig.lines.append(line)

    handles1, labels1 = ax1.get_legend_handles_labels()
    unique_handles1, unique_labels1 = [], []
    for handle, label in zip(handles1, labels1):
        if label not in unique_labels1:
            unique_handles1.append(handle)
            unique_labels1.append(label)
    ax1.legend(unique_handles1, unique_labels1, loc='lower left')

    handles2, labels2 = ax2.get_legend_handles_labels()
    unique_handles2, unique_labels2 = [], []
    for handle, label in zip(handles2, labels2):
        if label not in unique_labels2:
            unique_handles2.append(handle)
            unique_labels2.append(label)
    ax2.legend(unique_handles2, unique_labels2, loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')


    plt.show()