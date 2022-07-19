# tensorflow backend
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
# vae stuff
from chemvae_newTrain.vae_utils import VAEUtils
from chemvae_newTrain import mol_utils as mu
# import scientific py
import numpy as np
import pandas as pd
# rdkit stuff
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools
# plotting stuff
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import SVG, display

from chemvae_newTrain import mol_utils as mu
from chemvae_newTrain import hyperparameters
import random
import yaml
from chemvae_newTrain.models import load_encoder, load_decoder, load_property_predictor, load_varLayer
import numpy as np
import pandas as pd
import os
from chemvae_newTrain.mol_utils import fast_verify
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats, sparse
import numpy
from util import *
import anndata as ad

class samplefromNeighborsGenotype:
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



class samplefromNeighbors:
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



class VAEUtils(object):
	def __init__(self,
				 exp_file='exp.json',
				 encoder_file=None,
				 decoder_file=None,
				 varlayer_file = None, 
				 directory=None):
		# files
		if directory is not None:
			curdir = os.getcwd()
			os.chdir(os.path.join(curdir, directory))
			# exp_file = os.path.join(directory, exp_file)

		# load parameters
		self.params = hyperparameters.load_params(exp_file, False)
		if encoder_file is not None:
			self.params["encoder_weights_file"] = encoder_file
		if decoder_file is not None:
			self.params["decoder_weights_file"] = decoder_file

		if varlayer_file is not None:
			self.params["varlayer_weights_file"] = varlayer_file
		# char stuff
		chars = yaml.safe_load(open(self.params['char_file']))
		self.chars = chars
		self.params['NCHARS'] = len(chars)
		self.char_indices = dict((c, i) for i, c in enumerate(chars))
		self.indices_char = dict((i, c) for i, c in enumerate(chars))
		# encoder, decoder
		self.enc = load_encoder(self.params)
		self.dec = load_decoder(self.params)
		self.varlayer = load_varLayer(self.params)
		self.encode, self.decode, self.encode_sample, self.decode_sample = self.enc_dec_functions()
		self.data = None
		if self.params['do_prop_pred']:
			self.property_predictor = load_property_predictor(self.params)

		# Load data without normalization as dataframe
		df = pd.read_csv(self.params['data_file'])
		df.iloc[:, 0] = df.iloc[:, 0].str.strip()
		df = df[df.iloc[:, 0].str.len() <= self.params['MAX_LEN']]
		self.smiles = df.iloc[:, 0].tolist()
		if df.shape[1] > 1:
			self.data = df.iloc[:, 1:]

		self.estimate_estandarization()
		if directory is not None:
			os.chdir(curdir)
		return

	def estimate_estandarization(self):
		print('Standarization: estimating mu and std values ...', end='')
		# sample Z space

		smiles = self.random_molecules(size=50000)
		batch = 2500
		Z = np.zeros((len(smiles), self.params['hidden_dim']))
		Z_sample = np.zeros((len(smiles), self.params['hidden_dim']))
		for chunk in self.chunks(list(range(len(smiles))), batch):
			sub_smiles = [smiles[i] for i in chunk]
			one_hot = self.smiles_to_hot(sub_smiles)
			Z[chunk, :] = self.encode(one_hot, False)
			Z_sample[chunk, :] = self.encode_sample(one_hot, False)

		self.mu = np.mean(Z, axis=0)
		self.std = np.std(Z, axis=0)
		self.Z = self.standardize_z(Z)

		self.mu_sample = np.mean(Z_sample, axis=0)
		self.std_sample = np.std(Z_sample, axis=0)
		self.Z_sample = self.standardize_z_sample(Z_sample)


		print('done!')
		return

	def standardize_z(self, z):
		return (z - self.mu) / self.std

	def standardize_z_sample(self, z):
		return (z - self.mu_sample) / self.std_sample

	def unstandardize_z(self, z):
		return (z * self.std) + self.mu

	def unstandardize_z_sample(self, z):
		return (z * self.std_sample) + self.mu_sample

	def perturb_z(self, z, noise_norm, constant_norm=False):
		if noise_norm > 0.0:
			noise_vec = np.random.normal(0, 1, size=z.shape)
			noise_vec = noise_vec / np.linalg.norm(noise_vec)
			if constant_norm:
				return z + (noise_norm * noise_vec)
			else:
				noise_amp = np.random.uniform(
					0, noise_norm, size=(z.shape[0], 1))
				return z + (noise_amp * noise_vec)
		else:
			return z

	def smiles_distance_z(self, smiles, z0):
		x = self.smiles_to_hot(smiles)
		z_rep = self.encode(x)
		return np.linalg.norm(z0 - z_rep, axis=1)

	def prep_mol_df(self, smiles, z):
		df = pd.DataFrame({'smiles': smiles})
		sort_df = pd.DataFrame(df[['smiles']].groupby(
			by='smiles').size().rename('count').reset_index())
		df = df.merge(sort_df, on='smiles')
		df.drop_duplicates(subset='smiles', inplace=True)
		df = df[df['smiles'].apply(fast_verify)]
		if len(df) > 0:
			df['mol'] = df['smiles'].apply(mu.smiles_to_mol)
		if len(df) > 0:
			df = df[pd.notnull(df['mol'])]
		if len(df) > 0:
			df['distance'] = self.smiles_distance_z(df['smiles'], z)
			df['frequency'] = df['count'] / float(sum(df['count']))
			df = df[['smiles', 'distance', 'count', 'frequency', 'mol']]
			df.sort_values(by='distance', inplace=True)
			df.reset_index(drop=True, inplace=True)
		return df

	def z_to_smiles(self,
					z,
					decode_attempts=250,
					noise_norm=0.0,
					constant_norm=False,
					early_stop=None):
		if not (early_stop is None):
			Z = np.tile(z, (25, 1))
			Z = self.perturb_z(Z, noise_norm, constant_norm)
			X = self.decode(Z)
			smiles = self.hot_to_smiles(X, strip=True)
			df = self.prep_mol_df(smiles, z)
			if len(df) > 0:
				low_dist = df.iloc[0]['distance']
				if low_dist < early_stop:
					return df

		Z = np.tile(z, (decode_attempts, 1))
		Z = self.perturb_z(Z, noise_norm)
		X = self.decode(Z)
		smiles = self.hot_to_smiles(X, strip=True)
		df = self.prep_mol_df(smiles, z)
		return df

	def enc_dec_functions(self, standardized=True):
		print('Using standarized functions? {}'.format(standardized))
		if not self.params['do_tgru']:
			def decode(z, standardized=standardized):
				if standardized:
					return self.dec.predict(self.unstandardize_z(z))
				else:
					return self.dec.predict(z)

			def decode_sample(z, standardized=standardized):
				if standardized:
					return self.dec.predict(self.unstandardize_z_sample(z))
				else:
					return self.dec.predict(z)

		else:
			def decode(z, standardize=standardized):
				fake_shape = (z.shape[0], self.params[
					'MAX_LEN'], self.params['NCHARS'])
				fake_in = np.zeros(fake_shape)
				if standardize:
					return self.dec.predict([self.unstandardize_z(z), fake_in])
				else:
					return self.dec.predict([z, fake_in])

			def decode_sample(z, standardize=standardized):
				fake_shape = (z.shape[0], self.params[
					'MAX_LEN'], self.params['NCHARS'])
				fake_in = np.zeros(fake_shape)
				if standardize:
					return self.dec.predict([self.unstandardize_z_sample(z), fake_in])
				else:
					return self.dec.predict([z, fake_in])

		def encode(X, standardize=standardized):
			if standardize:
				return self.standardize_z(self.enc.predict(X)[0])
			else:
				return self.enc.predict(X)[0]

		def encode_sample(X, standardize = standardized):
			mean, middle = self.enc.predict(X)
			z_log_var, z_mean_log_var_output = self.varlayer.predict([mean, middle])
			epsilon = np.random.normal(size = (mean.shape[0], mean.shape[1]))
			z_samp = mean + np.exp(z_log_var / 2) * epsilon
			
			
			if standardize:
				return self.standardize_z_sample(z_samp)
			else:
				return z_samp

		return encode, decode, encode_sample, decode_sample

	# Now reports predictions after un-normalization.
	def predict_prop_Z(self, z, standardized=True):

		if standardized:
			z = self.unstandardize_z(z)

		# both regression and logistic
		if (('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0) and
				('logit_prop_tasks' in self.params) and (len(self.params['logit_prop_tasks']) > 0)):

			reg_pred, logit_pred = self.property_predictor.predict(z)
			if 'data_normalization_out' in self.params:
				df_norm = pd.read_csv(self.params['data_normalization_out'])
				reg_pred = reg_pred * \
					df_norm['std'].values + df_norm['mean'].values
			return reg_pred, logit_pred
		# regression only scenario
		elif ('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0):
			reg_pred = self.property_predictor.predict(z)
			if 'data_normalization_out' in self.params:
				df_norm = pd.read_csv(self.params['data_normalization_out'])
				reg_pred = reg_pred * \
					df_norm['std'].values + df_norm['mean'].values
			return reg_pred
		# logit only scenario
		else:
			logit_pred = self.property_predictor.predict(self.encode(z))
			return logit_pred

	# wrapper functions
	def predict_property_function(self):
		# Now reports predictions after un-normalization.
		def predict_prop(X):
			# both regression and logistic
			if (('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0) and
					('logit_prop_tasks' in self.params) and (len(self.params['logit_prop_tasks']) > 0)):
				reg_pred, logit_pred = self.property_predictor.predict(
					self.encode(X))
				if 'data_normalization_out' in self.params:
					df_norm = pd.read_csv(
						self.params['data_normalization_out'])
					reg_pred = reg_pred * \
						df_norm['std'].values + df_norm['mean'].values
				return reg_pred, logit_pred
			# regression only scenario
			elif ('reg_prop_tasks' in self.params) and (len(self.params['reg_prop_tasks']) > 0):
				reg_pred = self.property_predictor.predict(self.encode(X))
				if 'data_normalization_out' in self.params:
					df_norm = pd.read_csv(
						self.params['data_normalization_out'])
					reg_pred = reg_pred * \
						df_norm['std'].values + df_norm['mean'].values
				return reg_pred

			# logit only scenario
			else:
				logit_pred = self.property_predictor.predict(self.encode(X))
				return logit_pred

		return predict_prop

	def ls_sampler_w_prop(self, size=None, batch=2500, return_smiles=False):
		if self.data is None:
			print('use this sampler only for external property files')
			return

		cols = []
		if 'reg_prop_tasks' in self.params:
			cols += self.params['reg_prop_tasks']
		if 'logit_prop_tasks' in self.params:
			cols += self.params['logit_prop_tasks']
		idxs = self.random_idxs(size)
		smiles = [self.smiles[idx] for idx in idxs]
		data = [self.data.iloc[idx] for idx in idxs]
		Z = np.zeros((len(smiles), self.params['hidden_dim']))

		for chunk in self.chunks(list(range(len(smiles))), batch):
			sub_smiles = [smiles[i] for i in chunk]
			one_hot = self.smiles_to_hot(sub_smiles)
			Z[chunk, :] = self.encode(one_hot)

		if return_smiles:
			return Z, data, smiles

		return Z, data

	def smiles_to_hot(self, smiles, canonize_smiles=True, check_smiles=False):
		if isinstance(smiles, str):
			smiles = [smiles]

		if canonize_smiles:
			smiles = [mu.canon_smiles(s) for s in smiles]

		if check_smiles:
			smiles = mu.smiles_to_hot_filter(smiles, self.char_indices)

		p = self.params
		z = mu.smiles_to_hot(smiles,
							 p['MAX_LEN'],
							 p['PADDING'],
							 self.char_indices,
							 p['NCHARS'])
		return z

	def hot_to_smiles(self, hot_x, strip=False):
		smiles = mu.hot_to_smiles(hot_x, self.indices_char)
		if strip:
			smiles = [s.strip() for s in smiles]
		return smiles

	def random_idxs(self, size=None):
		if size is None:
			return [i for i in range(len(self.smiles))]
		else:
			return random.sample([i for i in range(len(self.smiles))], size)

	def random_molecules(self, size=None):
		if size is None:
			return self.smiles
		else:
			return random.sample(self.smiles, size)

	@staticmethod
	def chunks(l, n):
		"""Yield successive n-sized chunks from l."""
		for i in range(0, len(l), n):
			yield l[i:i + n]

if __name__ == "__main__":

	considered_epoch = [60, 70]

	for save_epoch in considered_epoch:

		for mean_opt in ['mean_rep', 'samp_rep']:

			for std_opt in [True, False]:

				# if save_epoch < 50 and std_opt:
				# 	continue

				std_note = 'nonStd'
				if std_opt:
					std_note = 'Std'

				path = './models/zinc/zinc_' + str(save_epoch) + 'epochs/'
				vae = VAEUtils(directory = path)
				hot_data = np.load('/nfs/turbo/umms-welchjd/hengshi/perturb_gan/chemical_vae-master_newTrain/OnehotData_188.npy')

				path_save = 'output_rfr2fid_chemvae_sample_' + str(save_epoch)  + 'epochs'
				path_data = '/nfs/turbo/umms-welchjd/hengshi/GAN/data/sciPlex/sciPlex3/'
				
				if not os.path.exists(path_save):
					os.makedirs(path_save, exist_ok = True)

				# import data
				adata = ad.read_h5ad(os.path.join(path_data, 'sciPlex3_whole_processed.h5ad'))
				metadata = adata.obs.copy()
				usedata = adata.X
				trt_whole_list = list(pd.read_csv(os.path.join(path_data, 'emb_named_PathwayLibrary.csv'))['treatment'])

				# data PC
				pca_data_50 = PCA(n_components=50, random_state = 42)
				pca_data_fit = pca_data_50.fit(usedata)


				# embedding
				considered_emb = ['chemvae_zinc', 'chemvae_zinc_property']
				fidscore_cal = fidscore()
				RFE = RandomForestError()
				
				emb = considered_emb[0]
				emb_file = 'emb_named_' + emb + '.csv'
				embdata = pd.read_csv(os.path.join(path_data, emb_file))

				# if t == 0:
				#   embdata_numpy = embdata.iloc[:, 2:12].values
				# elif t == 1:
				#   embdata_numpy  = embdata.iloc[:, 2:152].values
				# elif t == 2:
				#   embdata_numpy = embdata.iloc[:, 5:].values
				# elif t == 3:
				#   embdata_numpy = embdata.iloc[:, 5:].values
				# else:
				#   embdata_numpy = embdata.iloc[:, 2:770].values

				list_emb_trt = list(embdata['treatment'])
				#metadata['c_trt'] = metadata['cell_type'] + '_' + metadata['treatment']
				list_meta_ctype = list(metadata['cell_type'])
				list_meta_trt = list(metadata['treatment'])
				metadata['c_trt'] = [list_meta_ctype[i] + '_' + list_meta_trt[i] for i in range(len(list_meta_ctype))]

				list_c_trt = list(metadata['c_trt'])
				
				r2_distance_z = pd.DataFrame({'scheme':[], 'r2-KNN': [], 'r2-Random':[], 'n':[]})
				fid_distance_z = pd.DataFrame({'scheme':[], 'fid-KNN-dataPC': [], 'fid-Random-dataPC':[]})
				rf_distance_z = pd.DataFrame({'scheme':[], 'rf-KNN-dataPC': [],  'rf-Random-dataPC':[]})

				trt_cell_type_no = ['A549_S1628', 'K562_S1096', 'MCF7_S7259', 'MCF7_S1262', 'MCF7_S1259', 'MCF7_S7207']
				trt_type_no = trt_whole_list.copy()
				# KNN predictions

				for trt_type in trt_type_no:
					# indices for targe part

						# emb data
					
					if mean_opt == 'mean_rep':
						z_output_sample = None

						for i in range(len(hot_data)):
							
							z_1 = vae.encode(hot_data[[i]],  standardize = std_opt)
							if z_output_sample is None:
								z_output_sample = z_1
							else:
								z_output_sample = np.concatenate([z_output_sample, z_1], axis = 0)
						embdata_numpy = z_output_sample

					else:

						z_output_sample = None

						for i in range(len(hot_data)):
							
							z_1 = vae.encode_sample(hot_data[[i]],  standardize = std_opt)
							if z_output_sample is None:
								z_output_sample = z_1
							else:
								z_output_sample = np.concatenate([z_output_sample, z_1], axis = 0)
						embdata_numpy = z_output_sample


					idx_trt_type = [i for i in range(len(list_meta_trt)) if list_meta_trt[i] == trt_type]
					idx_nontrt_type = [i for i in range(len(list_meta_trt)) if list_meta_trt[i] != trt_type]

					trt_use = trt_type
					indice_trt = np.where(np.array(list_emb_trt) == trt_use)[0][0]
					embdata_obs = np.delete(embdata_numpy.copy(), indice_trt, axis = 0)
					
					list_emb_trtobs = list_emb_trt.copy()
					list_emb_trtobs.remove(trt_use)

					# nearest neighbors
					neigh = NearestNeighbors(n_neighbors = 5)
					neigh.fit(embdata_obs)

					distances, other_trts = neigh.kneighbors(embdata_numpy[[indice_trt]], 5, return_distance=True)
					# sampling with treatments
					samplerNN = samplefromNeighbors(distances, other_trts)
					idx_sample = samplerNN.samplingTrt(list_emb_trtobs, list_meta_trt, len(idx_trt_type))

					real_data, fake_data = usedata[idx_trt_type], usedata[idx_sample]

					# ### UMAP plots
					# figure_save_path = os.path.join(path_save, emb + '_sampled_fitted_' + trt_type + '.png')
					# samplerNN.PlotUMAP(real_data, fake_data, figure_save_path)

					### FID score 
					
					# fid_value = fidscore_cal.calculate_fid_score(real_data, fake_data)
					fid_value_d = fidscore_cal.calculate_fid_score(real_data, fake_data, pca_data_fit, if_dataPC = True)
					errors_d = RFE.fit_once(real_data, fake_data, pca_data_fit, if_dataPC = True, output_AUC = False)

					# ### R squared
					# control_sc = sc.AnnData(real_data, obs={"condition":["control"]*len(real_data)}, var={"var_names":[str(i) for i in list(range(real_data.shape[1]))]})
					# true_sc = sc.AnnData(real_data, obs={"condition":["True"]*len(real_data)},var={"var_names":[str(i) for i in list(range(real_data.shape[1]))]})
					# control_sc.obs['condition']  = ["control"] * len(control_sc)
					# true_sc.obs['condition']  = ["True"] * len(true_sc)

					# pred_adata = sc.AnnData(fake_data, obs={"condition":["pred"]*len(fake_data)}, var={"var_names":[str(i) for i in list(range(real_data.shape[1]))]})
					# all_adata = true_sc.concatenate(pred_adata)
					# all_adata = all_adata.concatenate(control_sc)
					# true_adata = true_sc.concatenate(control_sc)
					# true_adata.raw = true_adata

					# sc.tl.rank_genes_groups(true_adata, groupby="condition", method="wilcoxon")
					# diff_genes = true_adata.uns["rank_genes_groups"]["names"]["True"]


					# r2_value = compute_r2(all_adata, condition_key="condition",
					#                           axis_keys={
					#                               "x": "pred", "y": "True"},
					#                           gene_list=diff_genes[:10],
					#                           labels={"x": "predicted",
					#                                   "y": "ground truth"},
					#                           path_to_save="./pcon_metric_pred_real_ot/K562_from_A549.png",
					#                           show=True,
					#                           legend=False)
					# r2_distance_z = pd.concat([r2_distance_z, pd.DataFrame([[trt_type, r2_value]], columns = ['scheme', 'r2'])])
					# r2_distance_z.to_csv(os.path.join(path_save, emb + "_r2_distance_z.csv"))
					r2_value = fidscore_cal.calculate_r_square(real_data, fake_data)

					# # RF error
					# errors = list(RFE.fit(real_data, fake_data, output_AUC = False)['avg'])[0]
					# errors_d = list(RFE.fit(real_data, fake_data, pca_data_fit, if_dataPC = True, output_AUC = False)['avg'])[0]




					# random sample from the whole data
					idx_rsample = np.random.choice(idx_nontrt_type, len(idx_trt_type), replace = True)
					rfake_data = usedata[idx_rsample]

					errors_r_d = RFE.fit_once(real_data, rfake_data, pca_data_fit, if_dataPC = True, output_AUC = False)
					rf_distance_z = pd.concat([rf_distance_z, pd.DataFrame([[trt_type, errors_d, errors_r_d]], 
						columns = ['scheme', 'rf-KNN-dataPC','rf-Random-dataPC'])])
					rf_distance_z.to_csv(os.path.join(path_save, emb + "_" + mean_opt + "_" +  str(save_epoch) + "_" + std_note + "_rf_distance_z.csv"))

					# ### UMAP plots
					# figure_save_path = os.path.join(path_save, emb + '_randomsampled_fitted_' + trt_type + '.png')
					# samplerNN.PlotUMAP(real_data, rfake_data, figure_save_path)

					### FID score 
					#fid_value_r = fidscore_cal.calculate_fid_score(real_data, rfake_data)
					fid_value_r_d = fidscore_cal.calculate_fid_score(real_data, rfake_data, pca_data_fit, if_dataPC = True)
					fid_distance_z = pd.concat([fid_distance_z, pd.DataFrame([[trt_type, fid_value_d,  fid_value_r_d]], 
						columns = ['scheme',  'fid-KNN-dataPC', 'fid-Random-dataPC'])])
					fid_distance_z.to_csv(os.path.join(path_save, emb + "_" + mean_opt + "_" +  str(save_epoch) + "_" + std_note + "_fid_distance_z.csv"))


					# ### R squared
					# control_sc = sc.AnnData(real_data, obs={"condition":["control"]*len(real_data)}, var={"var_names":[str(i) for i in list(range(real_data.shape[1]))]})
					# true_sc = sc.AnnData(real_data, obs={"condition":["True"]*len(real_data)},var={"var_names":[str(i) for i in list(range(real_data.shape[1]))]})
					# control_sc.obs['condition']  = ["control"] * len(control_sc)
					# true_sc.obs['condition']  = ["True"] * len(true_sc)

					# pred_adata = sc.AnnData(rfake_data, obs={"condition":["pred"]*len(fake_data)}, var={"var_names":[str(i) for i in list(range(real_data.shape[1]))]})
					# all_adata = true_sc.concatenate(pred_adata)
					# all_adata = all_adata.concatenate(control_sc)
					# true_adata = true_sc.concatenate(control_sc)
					# true_adata.raw = true_adata

					# sc.tl.rank_genes_groups(true_adata, groupby="condition", method="wilcoxon")
					# diff_genes = true_adata.uns["rank_genes_groups"]["names"]["True"]


					# r2_value = compute_r2(all_adata, condition_key="condition",
					#                           axis_keys={
					#                               "x": "pred", "y": "True"},
					#                           gene_list=diff_genes[:10],
					#                           labels={"x": "predicted",
					#                                   "y": "ground truth"},
					#                           path_to_save="./pcon_metric_pred_real_ot/K562_from_A549.png",
					#                           show=True,
					#                           legend=False)
					# r2_distance_r = pd.concat([r2_distance_r, pd.DataFrame([[trt_type, r2_value]], columns = ['scheme', 'r2'])])
					# r2_distance_r.to_csv(os.path.join(path_save, emb + "_r2_distance_random.csv"))
					r2_value_r = fidscore_cal.calculate_r_square(real_data, rfake_data)
					r2_distance_z = pd.concat([r2_distance_z, pd.DataFrame([[trt_type, r2_value, r2_value_r, len(idx_trt_type)]], columns = ['scheme', 'r2-KNN', 'r2-Random', 'n'])])
					r2_distance_z.to_csv(os.path.join(path_save, emb + "_" + mean_opt + "_" +  str(save_epoch) + "_" + std_note + "_r2_distance_z.csv"))


					# # RF error
					# errors_r = list(RFE.fit(real_data, rfake_data, output_AUC = False)['avg'])[0]
					# errors_r_d = list(RFE.fit(real_data, rfake_data, pca_data_fit, if_dataPC = True, output_AUC = False)['avg'])[0]

					# rf_distance_z = pd.concat([rf_distance_z, pd.DataFrame([[trt_type, errors, errors_d, errors_r, errors_r_d]], 
					#   columns = ['scheme', 'rf-KNN', 'rf-KNN-dataPC','rf-Random', 'rf-Random-dataPC'])])
					# rf_distance_z.to_csv(os.path.join(path_save, emb + "_rf_distance_z.csv"))



