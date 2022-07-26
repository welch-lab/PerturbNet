#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import scvi

from anndata import AnnData
from scipy import linalg
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import stats, sparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class NormalizedRSquare:

	def __init__(self, largeCountData):
		self.calculate_largeData(largeCountData)

	def calculate_largeData(self, largeCountData):
		usedata = largeCountData.copy()
		usedata = usedata / usedata.sum(axis=1)[:, None]
		self.col_mu = usedata.mean(axis=0)
		self.col_std = usedata.std(axis=0)

	def calculate_r_square(self, real_data, fake_data):
		real_data_norm = real_data.copy()
		fake_data_norm = fake_data.copy()

		# normalize by row
		real_data_norm = real_data_norm / real_data_norm.sum(axis=1)[:, None]
		fake_data_norm = fake_data_norm / fake_data_norm.sum(axis=1)[:, None]
		# normalize by column
		real_data_norm = (real_data_norm - self.col_mu) / self.col_std
		fake_data_norm = (fake_data_norm - self.col_mu) / self.col_std

		x = np.average(fake_data_norm, axis=0)
		y = np.average(real_data_norm, axis=0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2

	def calculate_r_square_var(self, real_data, fake_data):
		real_data_norm = real_data.copy()
		fake_data_norm = fake_data.copy()

		# normalize by row
		real_data_norm = real_data_norm / real_data_norm.sum(axis=1)[:, None]
		fake_data_norm = fake_data_norm / fake_data_norm.sum(axis=1)[:, None]
		# normalize by column
		real_data_norm = (real_data_norm - self.col_mu) / self.col_std
		fake_data_norm = (fake_data_norm - self.col_mu) / self.col_std

		x = np.var(fake_data_norm, axis=0)
		y = np.var(real_data_norm, axis=0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2

class NormalizedRevisionRSquare:

	def __init__(self, largeCountData, targetSize = 1e4):
		
		self.targetSize = targetSize
		self.calculate_largeData(largeCountData)
		
	def calculate_largeData(self, largeCountData):
		usedata = largeCountData.copy()
		usedata = usedata / usedata.sum(axis = 1)[:, None] * self.targetSize
		usedata = np.log1p(usedata)

		self.col_mu = usedata.mean(axis = 0)
		self.col_std = usedata.std(axis = 0)

	def calculate_r_square(self, real_data, fake_data, max_value = 10):
		real_data_norm = real_data.copy()
		fake_data_norm = fake_data.copy()

		# normalize by row
		real_data_norm = real_data_norm / real_data_norm.sum(axis = 1)[:, None] * self.targetSize
		fake_data_norm = fake_data_norm / fake_data_norm.sum(axis = 1)[:, None] * self.targetSize
		
		real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
		real_data_norm = (real_data_norm - self.col_mu) / self.col_std
		fake_data_norm = (fake_data_norm - self.col_mu) / self.col_std

		real_data_norm[real_data_norm > max_value] = max_value
		fake_data_norm[fake_data_norm > max_value] = max_value

		x = np.average(fake_data_norm, axis = 0)
		y = np.average(real_data_norm, axis = 0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2, real_data_norm, fake_data_norm


class NormalizedRevisionRSquareLoad:

	def __init__(self,  col_mu, col_std, targetSize = 1e4):
		
		self.targetSize = targetSize
		self.col_mu = col_mu
		self.col_std = col_std
	
	def calculate_r_square(self, real_data, fake_data, max_value = 10):

		real_data_norm = real_data.copy()
		fake_data_norm = fake_data.copy()

		# normalize by row
		real_data_norm = real_data_norm / real_data_norm.sum(axis = 1)[:, None] * self.targetSize
		fake_data_norm = fake_data_norm / fake_data_norm.sum(axis = 1)[:, None] * self.targetSize
		
		real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
		real_data_norm = (real_data_norm - self.col_mu) / self.col_std
		fake_data_norm = (fake_data_norm - self.col_mu) / self.col_std

		real_data_norm[real_data_norm > max_value] = max_value
		fake_data_norm[fake_data_norm > max_value] = max_value

		x = np.average(fake_data_norm, axis = 0)
		y = np.average(real_data_norm, axis = 0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2, real_data_norm, fake_data_norm

class NormalizedRevisionRSquareVar:

	def __init__(self, norm_model):
		self.norm_model = norm_model

	def calculate_r_square_var(self, real_data, fake_data, max_value = 10):

		real_data_norm = real_data.copy()
		fake_data_norm = fake_data.copy()

		# normalize by row
		real_data_norm = real_data_norm / real_data_norm.sum(axis=1)[:, None] * self.norm_model.targetSize
		fake_data_norm = fake_data_norm / fake_data_norm.sum(axis=1)[:, None] * self.norm_model.targetSize

		real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(fake_data_norm)
		real_data_norm = (real_data_norm - self.norm_model.col_mu) / self.norm_model.col_std
		fake_data_norm = (fake_data_norm - self.norm_model.col_mu) / self.norm_model.col_std

		real_data_norm[real_data_norm > max_value] = max_value
		fake_data_norm[fake_data_norm > max_value] = max_value

		x = np.var(fake_data_norm, axis=0)
		y = np.var(real_data_norm, axis=0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2, real_data_norm, fake_data_norm

class fidscore:

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
		x = np.average(fake_data, axis = 0)
		y = np.average(real_data, axis = 0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2

	def calculate_r_square_var(self, real_data, fake_data):
		x = np.var(fake_data, axis = 0)
		y = np.var(real_data, axis = 0)
		m, b, r_value, p_value, std_err = stats.linregress(x, y)

		return r_value ** 2


class fidscore_scvi_extend(fidscore):

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


class RandomForestError:

	def __init__(self, n_folds = 5):
		super().__init__()
		self.rf = RandomForestClassifier(n_estimators = 1000,  random_state=42)
		self.pca_50 = PCA(n_components=50, random_state = 42)
		self.n_folds = n_folds

	def PrepareIndexes(self, pca_real, pca_fake):
		assert pca_real.shape[0] == pca_fake.shape[0]
		self.num_realize_gen = pca_real.shape[0]
		self.cat_t = ["1-training"] * self.num_realize_gen
		self.cat_g = ["2-generated"] * self.num_realize_gen
		self.cat_rf_gt = np.append(self.cat_g, self.cat_t)

		self.index_shuffle_mo =  list(range(self.num_realize_gen + self.num_realize_gen))
		np.random.shuffle(self.index_shuffle_mo)

		self.cat_rf_gt_s = self.cat_rf_gt[self.index_shuffle_mo]

		
		kf = KFold(n_splits = self.n_folds, random_state = 42)

		kf_cat_gt = kf.split(self.cat_rf_gt_s)
		self.train_in = np.array([])
		self.test_in = np.array([])
		self.train_cluster_in = np.array([])
		self.test_cluster_in = np.array([])

		j = 0
		for train_index, test_index in kf_cat_gt:
			self.train_in = np.append([self.train_in], [train_index])
			self.test_in = np.append([self.test_in], [test_index])
			self.train_cluster_in = np.append(self.train_cluster_in, np.repeat(j, len(train_index)))
			self.test_cluster_in = np.append(self.test_cluster_in, np.repeat(j, len(test_index)) )
			j+=1


	def fit(self, real_data, fake_data, pca_data_fit = None, if_dataPC = False, output_AUC = True, path_save = "."):

		if real_data.shape[0] <= 5 or fake_data.shape[0] <= 5:
			return -1
			
		all_data = np.concatenate([fake_data, real_data], axis = 0)
		if if_dataPC:
			pca_all = pca_data_fit.transform(all_data)
		else:
			pca_all = self.pca_50.fit(real_data).transform(all_data)
		pca_real, pca_fake = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]
		self.PrepareIndexes(pca_real, pca_fake)

		pca_gen_s = pca_all[self.index_shuffle_mo]

		vari = pca_gen_s # generated
		outc = self.cat_rf_gt_s
		# Binarize the output
		outc_1 = label_binarize(outc, classes=['', '1-training', '2-generated'])
		outc_1 = outc_1[:, 1:]
		n_classes = outc_1.shape[1]
		outc = np.array(outc)
		errors =  np.array([])
		for j in range(self.n_folds):
			train_index = [int(self.train_in[self.train_cluster_in == j][k]) for k in range(self.train_in[self.train_cluster_in == j].shape[0])]
			test_index = [int(self.test_in[self.test_cluster_in == j][k]) for k in range(self.test_in[self.test_cluster_in == j].shape[0])]
			X_train, X_test = vari[train_index], vari[test_index]
			y_train, y_test = outc[train_index], outc[test_index]
			y_test_1 = outc_1[test_index]
			self.rf.fit(X_train, y_train)
			predictions = self.rf.predict(X_test)
			errors = np.append(errors, np.mean((predictions != y_test)*1))
			
			if output_AUC:
				# AUC plots
				y_score_tr = self.rf.fit(X_train, y_train).predict_proba(X_test)

				fpr = dict()
				tpr = dict()
				roc_auc = dict()
				for k in range(n_classes):
					fpr[k], tpr[k], _ = roc_curve(y_test_1[:, k], y_score_tr[:, k])
					roc_auc[k] = auc(fpr[k], tpr[k])

				# Compute micro-average ROC curve and ROC area
				fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1.ravel(), y_score_tr.ravel())
				roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
				newfig = plt.figure()
				lw = 2
				plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
				plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
				plt.xlim([0.0, 1.0])
				plt.ylim([0.0, 1.05])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver operating characteristic')
				plt.legend(loc="lower right")
				plt.savefig(os.path.join(path_save, "gen_" + str(j) + "_fold_result.png"))
				plt.close(newfig)

		errors = np.append(errors, np.mean(errors))
		errors_pd = pd.DataFrame([errors], columns = ['1st', '2nd', '3rd' , '4th', '5th', 'avg'])
		

		return errors_pd

	def fit_once(self, real_data, fake_data, pca_data_fit = None, if_dataPC = False, output_AUC = True, path_save = "."):

		if real_data.shape[0] <= 5 or fake_data.shape[0] <= 5:
			return -1
			
		all_data = np.concatenate([fake_data, real_data], axis = 0)
		if if_dataPC:
			pca_all = pca_data_fit.transform(all_data)
		else:
			pca_all = self.pca_50.fit(real_data).transform(all_data)

		pca_real, pca_fake = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]
		self.PrepareIndexes(pca_real, pca_fake)

		pca_gen_s = pca_all[self.index_shuffle_mo]

		vari = pca_gen_s # generated
		outc = self.cat_rf_gt_s
		# Binarize the output
		outc_1 = label_binarize(outc, classes=['', '1-training', '2-generated'])
		outc_1 = outc_1[:, 1:]
		n_classes = outc_1.shape[1]
		outc = np.array(outc)

		
		j = 0

		train_index = [int(self.train_in[self.train_cluster_in == j][k]) for k in range(self.train_in[self.train_cluster_in == j].shape[0])]
		test_index = [int(self.test_in[self.test_cluster_in == j][k]) for k in range(self.test_in[self.test_cluster_in == j].shape[0])]
		X_train, X_test = vari[train_index], vari[test_index]
		y_train, y_test = outc[train_index], outc[test_index]
		y_test_1 = outc_1[test_index]
		self.rf.fit(X_train, y_train)
		predictions = self.rf.predict(X_test)
		errors = np.mean((predictions != y_test)*1)
		
		if output_AUC:
			# AUC plots
			y_score_tr = self.rf.fit(X_train, y_train).predict_proba(X_test)

			fpr = dict()
			tpr = dict()
			roc_auc = dict()
			for k in range(n_classes):
				fpr[k], tpr[k], _ = roc_curve(y_test_1[:, k], y_score_tr[:, k])
				roc_auc[k] = auc(fpr[k], tpr[k])

			# Compute micro-average ROC curve and ROC area
			fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1.ravel(), y_score_tr.ravel())
			roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
			newfig = plt.figure()
			lw = 2
			plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic')
			plt.legend(loc="lower right")
			plt.savefig(os.path.join(path_save, "gen_" + str(j) + "_fold_result.png"))
			plt.close(newfig)
		

		return errors

	def fit_full(self, real_data, fake_data, pca_data_fit=None, if_dataPC=False, output_AUC=True, path_save="."):
		"""
		directly fit the whole data and report the RF error
		"""

		if real_data.shape[0] <= 5 or fake_data.shape[0] <= 5:
			return -1

		all_data = np.concatenate([fake_data, real_data], axis=0)
		if if_dataPC:
			pca_all = pca_data_fit.transform(all_data)
		else:
			pca_all = self.pca_50.fit(real_data).transform(all_data)

		pca_real, pca_fake = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]
		self.PrepareIndexes(pca_real, pca_fake)

		pca_gen_s = pca_all[self.index_shuffle_mo]

		vari = pca_gen_s  # generated
		outc = self.cat_rf_gt_s
		# Binarize the output
		outc_1 = label_binarize(outc, classes=['', '1-training', '2-generated'])
		outc_1 = outc_1[:, 1:]
		n_classes = outc_1.shape[1]
		outc = np.array(outc)

		X_train = vari
		y_train = outc
		y_test_1 = outc_1
		self.rf.fit(X_train, y_train)
		predictions = self.rf.predict(X_train)
		errors = np.mean((predictions != y_train) * 1)

		if output_AUC:
			# AUC plots
			y_score_tr = self.rf.predict_proba(X_train)

			fpr = dict()
			tpr = dict()
			roc_auc = dict()
			for k in range(n_classes):
				fpr[k], tpr[k], _ = roc_curve(y_test_1[:, k], y_score_tr[:, k])
				roc_auc[k] = auc(fpr[k], tpr[k])

			# Compute micro-average ROC curve and ROC area
			fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1.ravel(), y_score_tr.ravel())
			roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
			newfig = plt.figure()
			lw = 2
			plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic')
			plt.legend(loc="lower right")
			plt.savefig(os.path.join(path_save, "gen_" + str(j) + "_fold_result.png"))
			plt.close(newfig)

		return errors


class LogisticRegError:

	def __init__(self, n_folds=5):
		super().__init__()
		self.rf = LogisticRegression(random_state=42)
		self.pca_50 = PCA(n_components=50, random_state=42)
		self.n_folds = n_folds

	def PrepareIndexes(self, pca_real, pca_fake):
		assert pca_real.shape[0] == pca_fake.shape[0]
		self.num_realize_gen = pca_real.shape[0]
		self.cat_t = ["1-training"] * self.num_realize_gen
		self.cat_g = ["2-generated"] * self.num_realize_gen
		self.cat_rf_gt = np.append(self.cat_g, self.cat_t)

		self.index_shuffle_mo = list(range(self.num_realize_gen + self.num_realize_gen))
		np.random.shuffle(self.index_shuffle_mo)

		self.cat_rf_gt_s = self.cat_rf_gt[self.index_shuffle_mo]

		kf = KFold(n_splits=self.n_folds, random_state=42)

		kf_cat_gt = kf.split(self.cat_rf_gt_s)
		self.train_in = np.array([])
		self.test_in = np.array([])
		self.train_cluster_in = np.array([])
		self.test_cluster_in = np.array([])

		j = 0
		for train_index, test_index in kf_cat_gt:
			self.train_in = np.append([self.train_in], [train_index])
			self.test_in = np.append([self.test_in], [test_index])
			self.train_cluster_in = np.append(self.train_cluster_in, np.repeat(j, len(train_index)))
			self.test_cluster_in = np.append(self.test_cluster_in, np.repeat(j, len(test_index)))
			j += 1

	def fit(self, real_data, fake_data, pca_data_fit=None, if_dataPC=False, output_AUC=True, path_save="."):

		if real_data.shape[0] <= 5 or fake_data.shape[0] <= 5:
			return -1

		all_data = np.concatenate([fake_data, real_data], axis=0)
		if if_dataPC:
			pca_all = pca_data_fit.transform(all_data)
		else:
			pca_all = self.pca_50.fit(real_data).transform(all_data)
		pca_real, pca_fake = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]
		self.PrepareIndexes(pca_real, pca_fake)

		pca_gen_s = pca_all[self.index_shuffle_mo]

		vari = pca_gen_s  # generated
		outc = self.cat_rf_gt_s
		# Binarize the output
		outc_1 = label_binarize(outc, classes=['', '1-training', '2-generated'])
		outc_1 = outc_1[:, 1:]
		n_classes = outc_1.shape[1]
		outc = np.array(outc)
		errors = np.array([])
		for j in range(self.n_folds):
			train_index = [int(self.train_in[self.train_cluster_in == j][k]) for k in
						   range(self.train_in[self.train_cluster_in == j].shape[0])]
			test_index = [int(self.test_in[self.test_cluster_in == j][k]) for k in
						  range(self.test_in[self.test_cluster_in == j].shape[0])]
			X_train, X_test = vari[train_index], vari[test_index]
			y_train, y_test = outc[train_index], outc[test_index]
			y_test_1 = outc_1[test_index]
			self.rf.fit(X_train, y_train)
			predictions = self.rf.predict(X_test)
			errors = np.append(errors, np.mean((predictions != y_test) * 1))

			if output_AUC:
				# AUC plots
				y_score_tr = self.rf.fit(X_train, y_train).predict_proba(X_test)

				fpr = dict()
				tpr = dict()
				roc_auc = dict()
				for k in range(n_classes):
					fpr[k], tpr[k], _ = roc_curve(y_test_1[:, k], y_score_tr[:, k])
					roc_auc[k] = auc(fpr[k], tpr[k])

				# Compute micro-average ROC curve and ROC area
				fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1.ravel(), y_score_tr.ravel())
				roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
				newfig = plt.figure()
				lw = 2
				plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
				plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
				plt.xlim([0.0, 1.0])
				plt.ylim([0.0, 1.05])
				plt.xlabel('False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver operating characteristic')
				plt.legend(loc="lower right")
				plt.savefig(os.path.join(path_save, "gen_" + str(j) + "_fold_result.png"))
				plt.close(newfig)

		errors = np.append(errors, np.mean(errors))
		errors_pd = pd.DataFrame([errors], columns=['1st', '2nd', '3rd', '4th', '5th', 'avg'])

		return errors_pd

	def fit_once(self, real_data, fake_data, pca_data_fit=None, if_dataPC=False, output_AUC=True, path_save="."):

		if real_data.shape[0] <= 5 or fake_data.shape[0] <= 5:
			return -1

		all_data = np.concatenate([fake_data, real_data], axis=0)
		if if_dataPC:
			pca_all = pca_data_fit.transform(all_data)
		else:
			pca_all = self.pca_50.fit(real_data).transform(all_data)

		pca_real, pca_fake = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]
		self.PrepareIndexes(pca_real, pca_fake)

		pca_gen_s = pca_all[self.index_shuffle_mo]

		vari = pca_gen_s  # generated
		outc = self.cat_rf_gt_s
		# Binarize the output
		outc_1 = label_binarize(outc, classes=['', '1-training', '2-generated'])
		outc_1 = outc_1[:, 1:]
		n_classes = outc_1.shape[1]
		outc = np.array(outc)

		j = 0

		train_index = [int(self.train_in[self.train_cluster_in == j][k]) for k in
					   range(self.train_in[self.train_cluster_in == j].shape[0])]
		test_index = [int(self.test_in[self.test_cluster_in == j][k]) for k in
					  range(self.test_in[self.test_cluster_in == j].shape[0])]
		X_train, X_test = vari[train_index], vari[test_index]
		y_train, y_test = outc[train_index], outc[test_index]
		y_test_1 = outc_1[test_index]
		self.rf.fit(X_train, y_train)
		predictions = self.rf.predict(X_test)
		errors = np.mean((predictions != y_test) * 1)

		if output_AUC:
			# AUC plots
			y_score_tr = self.rf.fit(X_train, y_train).predict_proba(X_test)

			fpr = dict()
			tpr = dict()
			roc_auc = dict()
			for k in range(n_classes):
				fpr[k], tpr[k], _ = roc_curve(y_test_1[:, k], y_score_tr[:, k])
				roc_auc[k] = auc(fpr[k], tpr[k])

			# Compute micro-average ROC curve and ROC area
			fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1.ravel(), y_score_tr.ravel())
			roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
			newfig = plt.figure()
			lw = 2
			plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic')
			plt.legend(loc="lower right")
			plt.savefig(os.path.join(path_save, "gen_" + str(j) + "_fold_result.png"))
			plt.close(newfig)

		return errors

	def fit_full(self, real_data, fake_data, pca_data_fit=None, if_dataPC=False, output_AUC=True, path_save="."):
		"""
		directly fit the whole data and report the RF error
		"""

		if real_data.shape[0] <= 5 or fake_data.shape[0] <= 5:
			return -1

		all_data = np.concatenate([fake_data, real_data], axis=0)
		if if_dataPC:
			pca_all = pca_data_fit.transform(all_data)
		else:
			pca_all = self.pca_50.fit(real_data).transform(all_data)

		pca_real, pca_fake = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]
		self.PrepareIndexes(pca_real, pca_fake)

		pca_gen_s = pca_all[self.index_shuffle_mo]

		vari = pca_gen_s  # generated
		outc = self.cat_rf_gt_s
		# Binarize the output
		outc_1 = label_binarize(outc, classes=['', '1-training', '2-generated'])
		outc_1 = outc_1[:, 1:]
		n_classes = outc_1.shape[1]
		outc = np.array(outc)

		X_train = vari
		y_train = outc
		y_test_1 = outc_1
		self.rf.fit(X_train, y_train)
		predictions = self.rf.predict(X_train)
		errors = np.mean((predictions != y_train) * 1)

		if output_AUC:
			# AUC plots
			y_score_tr = self.rf.predict_proba(X_train)

			fpr = dict()
			tpr = dict()
			roc_auc = dict()
			for k in range(n_classes):
				fpr[k], tpr[k], _ = roc_curve(y_test_1[:, k], y_score_tr[:, k])
				roc_auc[k] = auc(fpr[k], tpr[k])

			# Compute micro-average ROC curve and ROC area
			fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1.ravel(), y_score_tr.ravel())
			roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
			newfig = plt.figure()
			lw = 2
			plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic')
			plt.legend(loc="lower right")
			plt.savefig(os.path.join(path_save, "gen_" + str(j) + "_fold_result.png"))
			plt.close(newfig)

		return errors


class Standardize:
	
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
	def __init__(self, mu, std, device):
		self.mu = mu
		self.std = std
		self.device = device

	def standardize_z(self, z):
		return (z - self.mu) / self.std

	def standardize_z_torch(self, z):
		return (z - torch.tensor(self.mu).float().to(self.device)) / torch.tensor(self.std).float().to(self.device)


class ConcatDatasetWithIndices(torch.utils.data.Dataset):

	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, i):
		return tuple([d[i] for d in self.datasets] + [i])

	def __len__(self):
		return min(len(d) for d in self.datasets)



class SaveEvaluationResults:
	"""
	saving evaluation results
	"""
	def __init__(self, method_d, method_r):

		super().__init__()

		self.method_d = method_d
		self.method_r = method_r
		self.r2_distance_z = pd.DataFrame({'scheme':[], 'n':[], 'r2-' + method_d: [], 'r2-' + method_r:[]})
		self.fid_distance_z = pd.DataFrame({'scheme':[], 'n':[], 'fid-' + method_d: [], 'fid-' + method_r:[]})
		self.rf_distance_z = pd.DataFrame({'scheme':[], 'n':[], 'rf-' + method_d: [], 'rf-' + method_r:[]})
		self.fid_scvi_distance_z_mu = pd.DataFrame({'scheme':[], 'n':[], 'fid-' + method_d: [], 'fid-' + method_r:[]})
		self.fid_scvi_distance_z_sample = pd.DataFrame({'scheme':[], 'n':[], 'fid-' + method_d: [], 'fid-' + method_r:[]})

	def update(self, trt_type, n_sample, 
			   r2_value_d, r2_value_r, 
			   fid_value_d, fid_value_r, 
			   errors_d, errors_r, 
			   fid_value_d_scvi_sample, fid_value_r_scvi_sample, 
			   fid_value_d_scvi_mu, fid_value_r_scvi_mu):

		methods_dr = [self.method_d, self.method_r]

		self.r2_distance_z = pd.concat([self.r2_distance_z, pd.DataFrame([[trt_type, n_sample, r2_value_d, r2_value_r]], 
			columns = ['scheme', 'n'] + ['r2-' + m for m in methods_dr])])
		self.rf_distance_z = pd.concat([self.rf_distance_z, pd.DataFrame([[trt_type, n_sample, errors_d, errors_r]], 
			columns = ['scheme', 'n'] + ['rf-' + m for m in methods_dr])])
		
		self.fid_distance_z = pd.concat([self.fid_distance_z, pd.DataFrame([[trt_type, n_sample, fid_value_d, fid_value_r]], 
			columns = ['scheme', 'n'] + ['fid-' + m for m in methods_dr])])
		self.fid_scvi_distance_z_mu = pd.concat([self.fid_scvi_distance_z_mu, pd.DataFrame([[trt_type, n_sample, fid_value_d_scvi_mu, fid_value_r_scvi_mu]], 
			columns = ['scheme', 'n'] + ['fid-' + m for m in methods_dr])])
		self.fid_scvi_distance_z_sample = pd.concat([self.fid_scvi_distance_z_sample, pd.DataFrame([[trt_type, n_sample, fid_value_d_scvi_sample, fid_value_r_scvi_sample]], 
			columns = ['scheme', 'n'] + ['fid-' + m for m in methods_dr])])

	def saveToCSV(self, path_save, file_save, indice_start = 0, indice_end = None):

		if indice_end is None:
			indice_end = self.r2_distance_z.shape[0]

		self.r2_distance_z.iloc[indice_start:indice_end, :].to_csv(os.path.join(path_save, file_save + "_r2_distance_z.csv"))
		self.rf_distance_z.iloc[indice_start:indice_end, :].to_csv(os.path.join(path_save, file_save + "_rf_distance_z.csv"))
		
		self.fid_distance_z.iloc[indice_start:indice_end, :].to_csv(os.path.join(path_save, file_save + "_fid_distance_z.csv"))
		self.fid_scvi_distance_z_mu.iloc[indice_start:indice_end, :].to_csv(os.path.join(path_save, file_save + "_fid_distance_z_scvi_mu.csv"))
		self.fid_scvi_distance_z_sample.iloc[indice_start:indice_end, :].to_csv(os.path.join(path_save, file_save + "_fid_distance_z_scvi_sample.csv"))


