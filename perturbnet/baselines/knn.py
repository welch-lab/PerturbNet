#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import umap
from plotnine import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class samplefromNeighbors:
	"""
	KNN sampling module 
	"""
	def __init__(self, distances, other_trts):
		super().__init__()
		self.distances = distances
		self.other_trts = other_trts
		self.softmax()
		self.pca_50 = PCA(n_components=50, random_state = 42)

	def softmax(self):
		prob_array = np.exp(-self.distances)
		prob_sum = prob_array.sum()
		prob_array /= prob_sum
		self.prob_array  = prob_array

	def samplingCTrt(self, list_trt, cell_type, list_c_trt, n_sample = 300):

		needed_trts = np.random.choice(self.other_trts.squeeze(), n_sample,  replace = True, p = self.prob_array.squeeze())
		trt_sample_array = np.array(list_trt)[needed_trts]
		trt_sample_pd = pd.Series(trt_sample_array).value_counts()

		idx_sample = None
		for t in range(len(list(trt_sample_pd.keys()))):
			trt_sample = list(trt_sample_pd.keys())[t]
			trt_sample_count = trt_sample_pd.values[t]
			ctrt_sample = cell_type + "_" + trt_sample

			idx_sample_type = [i for i in range(len(list_c_trt)) if list_c_trt[i] == ctrt_sample]
			idx_ctrt_sample = np.random.choice(idx_sample_type, trt_sample_count, replace = True)

			if idx_sample is None:
				idx_sample = idx_ctrt_sample
			else:
				idx_sample = np.append(idx_sample, idx_ctrt_sample)

		return idx_sample

	def samplingTrt(self, list_trt, list_data_trt, n_sample = 300):

		needed_trts = np.random.choice(self.other_trts.squeeze(), n_sample,  replace = True, p = self.prob_array.squeeze())
		trt_sample_array = np.array(list_trt)[needed_trts]
		trt_sample_pd = pd.Series(trt_sample_array).value_counts()

		idx_sample = None
		for t in range(len(list(trt_sample_pd.keys()))):
			trt_sample = list(trt_sample_pd.keys())[t]
			trt_sample_count = trt_sample_pd.values[t]
			ctrt_sample = trt_sample

			idx_sample_type = [i for i in range(len(list_data_trt)) if list_data_trt[i] == ctrt_sample]
			idx_ctrt_sample = np.random.choice(idx_sample_type, trt_sample_count, replace = True)

			if idx_sample is None:
				idx_sample = idx_ctrt_sample
			else:
				idx_sample = np.append(idx_sample, idx_ctrt_sample)

		return idx_sample

	def PlotUMAP(self, real_data, fake_data, path_file_save):

		all_data = np.concatenate([fake_data, real_data], axis = 0)
		pca_all = self.pca_50.fit(real_data).transform(all_data)
		pca_result_real = pca_all[fake_data.shape[0]:]

		cat_t = ["1-Real"] * real_data.shape[0]
		cat_g = ["2-KNN-Sampled"] * fake_data.shape[0]
		cat_rf_gt = np.append(cat_g, cat_t)

		trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(pca_result_real)

		X_embedded_pr = trans.transform(pca_all)
		df_tsne_pr = X_embedded_pr.copy()
		df_tsne_pr = pd.DataFrame(df_tsne_pr)
		df_tsne_pr['x-umap'] = X_embedded_pr[:,0]
		df_tsne_pr['y-umap'] = X_embedded_pr[:,1]
		df_tsne_pr['category'] = cat_rf_gt
			
		chart_pr = ggplot(df_tsne_pr, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
		+ geom_point(size=0.5, alpha = 0.8) \
		+ ggtitle("UMAP dimensions")
		chart_pr.save(path_file_save, width=12, height=8, dpi=144)


class samplefromNeighborsCINN:
	"""
	KNN sampling module within a constrained list
	"""
	def __init__(self, distances, other_trts):
		super().__init__()
		self.distances = distances
		self.other_trts = other_trts
		self.softmax()
		self.pca_50 = PCA(n_components=50, random_state=42)

	def softmax(self):
		prob_array = np.exp(-self.distances)
		prob_sum = prob_array.sum()
		prob_array /= prob_sum
		self.prob_array = prob_array

	def samplingTrt(self, data_onehot, n_sample=300):

		needed_trts = np.random.choice(self.other_trts.squeeze(), n_sample, replace=True, p=self.prob_array.squeeze())
		onehot_data = data_onehot[needed_trts]

		return onehot_data

	def samplingTrtList(self, data_onehot, list_trt, list_data_trt,  n_sample=300):
		list_trt_pd = pd.Series(list_trt)
		list_data_trt_pd = pd.Series(list_data_trt)

		indices = list_data_trt_pd.map(lambda x: np.where(list_trt_pd == x)[0][0]).tolist()
		onehot_data = data_onehot[indices]

		return onehot_data





class samplefromNeighborsGenotype:
	"""
	KNN sampling module for genetic perturbations, especially for these with multiple target genes
	"""
	def __init__(self, distances, other_trts):
		super().__init__()
		self.distances = distances
		self.other_trts = other_trts
		self.softmax()
		self.pca_50 = PCA(n_components=50, random_state = 42)


	def softmax(self):
		prob_array = np.exp(-self.distances)
		prob_sum = prob_array.sum()
		prob_array /= prob_sum
		self.prob_array  = prob_array

	def samplingCTrt(self, list_trt, cell_type, list_c_trt, n_sample = 300):

		needed_trts = np.random.choice(self.other_trts.squeeze(), n_sample,  replace = True, p = self.prob_array.squeeze())
		trt_sample_array = np.array(list_trt)[needed_trts]
		trt_sample_pd = pd.Series(trt_sample_array).value_counts()

		idx_sample = None
		for t in range(len(list(trt_sample_pd.keys()))):
			trt_sample = list(trt_sample_pd.keys())[t]
			trt_sample_count = trt_sample_pd.values[t]
			ctrt_sample = cell_type + "_" + trt_sample

			idx_sample_type = [i for i in range(len(list_c_trt)) if list_c_trt[i] == ctrt_sample]
			idx_ctrt_sample = np.random.choice(idx_sample_type, trt_sample_count, replace = True)

			if idx_sample is None:
				idx_sample = idx_ctrt_sample
			else:
				idx_sample = np.append(idx_sample, idx_ctrt_sample)

		return idx_sample

	def samplingTrt(self, list_trt, list_data_trt, n_sample = 300):

		needed_trts = np.random.choice(self.other_trts.squeeze(), n_sample,  replace = True, p = self.prob_array.squeeze())
		trt_sample_array = np.array(list_trt)[needed_trts]
		trt_sample_pd = pd.Series(trt_sample_array).value_counts()

		idx_sample = None
		for t in range(len(list(trt_sample_pd.keys()))):
			trt_sample = list(trt_sample_pd.keys())[t]
			trt_sample_count = trt_sample_pd.values[t]
			ctrt_sample = trt_sample
			trt_sample1, trt_sample2 = trt_sample.split('/')
			ctrt_sample_other = trt_sample2  + '/' + trt_sample1

			idx_sample_type = [i for i in range(len(list_data_trt)) if list_data_trt[i] in [ctrt_sample, ctrt_sample_other]]
			idx_ctrt_sample = np.random.choice(idx_sample_type, trt_sample_count, replace = True)

			if idx_sample is None:
				idx_sample = idx_ctrt_sample
			else:
				idx_sample = np.append(idx_sample, idx_ctrt_sample)

		return idx_sample

	def PlotUMAP(self, real_data, fake_data, path_file_save):

		all_data = np.concatenate([fake_data, real_data], axis = 0)
		pca_all = self.pca_50.fit(real_data).transform(all_data)
		pca_result_real = pca_all[fake_data.shape[0]:]

		cat_t = ["1-Real"] * real_data.shape[0]
		cat_g = ["2-KNN-Sampled"] * fake_data.shape[0]
		cat_rf_gt = np.append(cat_g, cat_t)

		trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(pca_result_real)

		X_embedded_pr = trans.transform(pca_all)
		df_tsne_pr = X_embedded_pr.copy()
		df_tsne_pr = pd.DataFrame(df_tsne_pr)
		df_tsne_pr['x-umap'] = X_embedded_pr[:,0]
		df_tsne_pr['y-umap'] = X_embedded_pr[:,1]
		df_tsne_pr['category'] = cat_rf_gt
			
		chart_pr = ggplot(df_tsne_pr, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
		+ geom_point(size=0.5, alpha = 0.8) \
		+ ggtitle("UMAP dimensions")
		chart_pr.save(path_file_save, width=12, height=8, dpi=144)


class samplefromNeighborsGenotypeCINN:
	"""
	KNN sampling module within a constrained list for genetic perturbations, 
	especially for these with multiple target genes
	"""
	def __init__(self, distances, other_trts):
		super().__init__()
		self.distances = distances
		self.other_trts = other_trts
		self.softmax()
		self.pca_50 = PCA(n_components=50, random_state=42)

	def softmax(self):
		prob_array = np.exp(-self.distances)
		prob_sum = prob_array.sum()
		prob_array /= prob_sum
		self.prob_array = prob_array


	def samplingTrt(self, data_onehot, n_sample=300):

		needed_trts = np.random.choice(self.other_trts.squeeze(), n_sample, replace=True, p=self.prob_array.squeeze())
		onehot_data = data_onehot[needed_trts]

		return onehot_data

	def samplingTrtList(self, data_onehot, perturbToOnehotLib, list_data_trt,  n_sample=300):
		list_data_trt_pd = pd.Series(list_data_trt)

		indices = list_data_trt_pd.map(lambda x: perturbToOnehotLib[x]).tolist()
		onehot_data = data_onehot[indices]

		return onehot_data
