#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
from random import shuffle
from scipy import sparse

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd

import umap
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from plotnine import * 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def tf_log(value):
	"""
	tensorflow logrithmic 
	"""
	return tf.log(value + 1e-16)
	
def logsumexp(value,  dim = None, keepdims = False):
	"""
	calculate q(Z) = sum_{X}[sum_{j}q(Zj|X)] and q(Zj) = sum_{X}[q(Zj|X)]
	"""
	if dim is not None:
		m = tf.reduce_max(value, axis = dim, keepdims = True)
		value0 = tf.subtract(value, m)
		if keepdims is False:
			m = tf.squeeze(m, dim)
		return tf.add(m, tf_log(tf.reduce_sum(tf.exp(value0), axis = dim, keepdims = keepdims)))

	else:
		m = tf.reduce_max(value)
		sum_exp = tf.reduce_sum(tf.exp(tf.subtract(value, m)))  
		return tf.add(m, tf_log(sum_exp))


def total_correlation(marginal_entropies, joint_entropies):
	"""
	calculate total correlation from the marginal and joint entropies
	"""
	return tf.reduce_sum(marginal_entropies) - tf.reduce_sum(joint_entropies)


class MetricVisualize:
	"""
	Generation measure class
	"""
	def __init__(self):
		super().__init__
		self.pca_50 = PCA(n_components=50, random_state = 42)
		self.rf = RandomForestClassifier(n_estimators = 1000,  random_state=42)

	def CorrelationDistance(self, data):
		"""
		calculate correlation distance within a dataset
		"""
		return round(np.median(pdist(data, metric='correlation')), 3)


	def FIDScore(self, real_data, fake_data, pca_data_fit = None, if_dataPC = False):
		"""
		calculate Frechet inception distance between real and fake data on the PC space
		"""

		all_data = np.concatenate([fake_data, real_data], axis = 0)

		if if_data_PC:
			pca_all = pca_data_fit.transform(all_data)
		else:
			pca_all = self.pca_50.fit(real_data).transform(all_data)

		pca_real, pca_fake = pca_all[fake_data.shape[0]:], pca_all[:fake_data.shape[0]]

		FIDval = calculate_fid_score(pca_fake, pca_real)

		return FIDval

	def InceptionScore(self, real_data, real_cell_type, target_data):
		"""
		calculate inception score of target data based on the cell type random forest classifier
		on the real data
		"""

		rf_fit = self.rf.fit(real_data, real_cell_type)
		data_score = rf_fit.predict_proba(target_data)

		meanScore, stdScore = preds2score(data_score, data_score.mean(axis = 0), splits = 3)

		return meanScore, stdScore



	def umapPlot(self, real_data, fake_data, path_file_save = None):
		"""
		UMAP plot of real and fake data
		"""
		all_data = np.concatenate([fake_data, real_data], axis = 0)
		pca_all = self.pca_50.fit(real_data).transform(all_data)
		pca_result_real = pca_all[fake_data.shape[0]:]

		cat_t = ["1-Real"] * real_data.shape[0]
		cat_g = ["2-Fake"] * fake_data.shape[0]
		cat_rf_gt = np.append(cat_g, cat_t)

		trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(pca_result_real)

		X_embedded_pr = trans.transform(pca_all)
		df_tsne_pr = X_embedded_pr.copy()
		df_tsne_pr = pd.DataFrame(df_tsne_pr)
		df_tsne_pr['x-umap'] = X_embedded_pr[:,0]
		df_tsne_pr['y-umap'] = X_embedded_pr[:,1]
		df_tsne_pr['category'] = cat_rf_gt
			
		chart_pr = ggplot(df_tsne_pr, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
		+ geom_point(size=0.5, alpha = 0.5) \
		+ ggtitle("UMAP dimensions")
		
		if path_file_save is not None:
			chart_pr.save(path_file_save, width=12, height=8, dpi=144)
		
		return chart_pr

	def umapPlotByCat(self, pca_data_fit, data, data_category, path_file_save = None):
		"""
		UMAP plot of data colored by categories. It involves a PCA procedure
		"""
		pca_data = pca_data_fit.transform(data)

		trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(pca_result_real)

		X_embedded_pr = trans.transform(pca_data)
		df_tsne_pr = X_embedded_pr.copy()
		df_tsne_pr = pd.DataFrame(df_tsne_pr)
		df_tsne_pr['x-umap'] = X_embedded_pr[:,0]
		df_tsne_pr['y-umap'] = X_embedded_pr[:,1]
		df_tsne_pr['category'] = data_category
			
		chart_pr = ggplot(df_tsne_pr, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
		+ geom_point(size=0.5, alpha = 0.5) \
		+ ggtitle("UMAP dimensions")

		if path_file_save is not None:
			chart_pr.save(path_file_save, width=12, height=8, dpi=144)
		
		return chart_pr


	def umapPlotPurelyByCat(self, umap_data, data_category, path_file_save = None):
		"""
		UMAP plot of data colored by categories. It directly has the UMAP data as an input. 
		"""
		df_tsne_pr = umap_data.copy()
		df_tsne_pr = pd.DataFrame(df_tsne_pr)
		df_tsne_pr['x-umap'] = umap_data[:,0]
		df_tsne_pr['y-umap'] = umap_data[:,1]
		df_tsne_pr['category'] = data_category
			
		chart_pr = ggplot(df_tsne_pr, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
		+ geom_point(size=0.5, alpha = 0.5) \
		+ ggtitle("UMAP dimensions")
		
		if path_file_save is not None:
			chart_pr.save(path_file_save, width=12, height=8, dpi=144)
		return chart_pr

	
	def umapPlotPurelyByCatHighQuality(self, umap_data, xlab_showname, ylab_showname, data_category, 
									   nrowlegend = 7, size = 5, alpha = 1, legend_title = 'UMAP Plot', 
									   path_file_save = None):
		"""
		high-quality UMAP plot of umap data by categories.  
		"""
		df_tsne_pr = umap_data.copy()
		df_tsne_pr = pd.DataFrame(df_tsne_pr)
		df_tsne_pr['x-umap'] = umap_data[:,0]
		df_tsne_pr['y-umap'] = umap_data[:,1]
		df_tsne_pr['category'] = data_category
			
		chart_pr = ggplot(df_tsne_pr, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
		+ geom_point(size = size, alpha = alpha) + labs(x = xlab_showname, y = ylab_showname) \
		+ geom_abline(intercept = 0 , slope = 1, size=1, linetype="dashed", color="black") \
		+ xlim(0, 1) + ylim(0, 1) + theme_bw() \
		+ theme(panel_background = element_rect(fill='white'),
				title = element_text(size = 25), 
				axis_title_x = element_text(size = 25), 
				axis_title_y = element_text(size = 25),
				axis_text_x = element_text(size = 15), 
				axis_text_y = element_text(size = 15),
				legend_title = element_text(size = 20), 
				legend_text = element_text(size = 20), 
				axis_ticks_major_y = element_blank(),
				axis_ticks_major_x = element_blank(), 
				panel_grid = element_blank()) \
		+ ggtitle(legend_title) \
		+ guides(colour = guide_legend(nrow=nrowlegend, override_aes={"size": 10}))

		if path_file_save is not None:
			chart_pr.save(path_file_save, width=12, height=8, dpi=144)
		return chart_pr

	def latentHistPlot(self, z_data,  path_file_save = None):
		"""
		Plot of histograms 
		"""

		dict_use = {}
		
		for h in range(z_data.shape[1]):

			dict_use["Var " + str(h+1)] = h + 1

		newfig = plt.figure(figsize=[20,16])
		for m in range(z_data.shape[1]):
			name_i = list(dict_use.keys())[m]
			num_i = dict_use[name_i]
			ax1 = newfig.add_subplot(4, 3, t + 1)
			weights = np.ones_like(z_data[:,m])/float(len(z_data[:,m]))
			ax1.hist(z_data[:,m], bins = 100, weights = weights, alpha = 0.5)
			ax1.set_title(name_i)

		if path_file_save is not None:
			newfig.savefig(path_file_save)

	def latentColorPlot(self, z_data, umapData, path_file_save = None):
		"""
		UMAP plots by latent values
		"""

		dict_use = {}
		for h in range(z_data.shape[1]):
			dict_use["Var " + str(h+1)] = h + 1     

		# mapped
		newfig = plt.figure(figsize=[20,16])
		for m in range(len(dict_use)):
			name_i = list(dict_use.keys())[m]
			num_i = dict_use[name_i]
			ax1 = newfig.add_subplot(4, 3,num_i)
			cb1 = ax1.scatter(umapData['x-umap'], umapData['y-umap'], s= 1, c = z_data[:, m], cmap= "plasma")
			ax1.set_title(name_i)
		
		if path_file_save is not None:
			newfig.savefig(path_file_save)




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

		
		kf = KFold(n_splits = self.n_folds, random_state = 42, shuffle=True)

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

		if real_data.shape[0] <= 5:
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

		if real_data.shape[0] <= 5:
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