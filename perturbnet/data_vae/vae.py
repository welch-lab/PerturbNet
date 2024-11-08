#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import logging

import numpy as np
import pandas as pd
from scipy import sparse

from random import sample
import tensorflow as tf
from tensorflow import distributions as ds

from .util import *
from tqdm import tqdm

log = logging.getLogger(__file__)

class VAE:
	"""
	General VAE (beta = 0) and beta-TCVAE class 
	"""
	def __init__(self, num_cells_train, x_dimension, z_dimension = 10, **kwargs):
		tf.compat.v1.reset_default_graph()
		self.num_cells_train = num_cells_train
		self.x_dim = x_dimension
		self.z_dim = z_dimension
		self.learning_rate = kwargs.get("learning_rate", 1e-3)
		self.dropout_rate = kwargs.get("dropout_rate", 0.2)
		self.beta = kwargs.get("beta", 0.0)
		self.alpha = kwargs.get("alpha", 1.0)
		self.inflate_to_size1 = kwargs.get("inflate_size_1", 256)
		self.inflate_to_size2 = kwargs.get("inflate_size_2", 512)
		self.disc_internal_size2 = kwargs.get("disc_size_2", 512)
		self.disc_internal_size3 = kwargs.get("disc_size_3", 256)
		self.if_BNTrainingMode = kwargs.get("BNTrainingMode", True)
		self.is_training = tf.placeholder(tf.bool, name = "training_flag")
		
		self.init_w = tf.contrib.layers.xavier_initializer()
		self.device = kwargs.get("device",  '/device:GPU:0')

		with tf.device(self.device):
			self.x = tf.placeholder(tf.float32, shape = [None, self.x_dim], name = "data")
			self.z = tf.placeholder(tf.float32, shape = [None, self.z_dim], name = "latent")
			self.create_network()
			self.loss_function()
		
		config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.6
		self.sess = tf.Session(config = config)

		self.saver = tf.train.Saver(max_to_keep = 1)
		self.init = tf.global_variables_initializer().run(session = self.sess)
		self.train_loss = []
		self.valid_loss = []
		self.training_time = 0.0

	def encoder(self):
		""" 
		encoder of VAE
		"""
		with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):        
			en_dense2 = tf.layers.dense(inputs = self.x, units = self.inflate_to_size2, activation = None, 
				kernel_initializer = self.init_w)
			en_dense2 = tf.layers.batch_normalization(en_dense2, training = self.is_training)
			en_dense2 = tf.nn.leaky_relu(en_dense2)
			en_dense2 = tf.layers.dropout(en_dense2, self.dropout_rate, training = self.is_training)

			en_dense3 = tf.layers.dense(inputs = en_dense2, units = self.inflate_to_size1, activation = None, 
				kernel_initializer = self.init_w)
			en_dense3 = tf.layers.batch_normalization(en_dense3, training = self.is_training)
			en_dense3 = tf.nn.relu(en_dense3)
			en_dense3 = tf.layers.dropout(en_dense3, self.dropout_rate, training = self.is_training)
			
			en_loc = tf.layers.dense(inputs=en_dense3, units= self.z_dim, activation=None, kernel_initializer = self.init_w)
			en_scale = tf.layers.dense(inputs = en_dense3, units= self.z_dim, activation=None, kernel_initializer = self.init_w)
			en_scale = tf.nn.softplus(en_scale)
			return en_loc, en_scale

	def decoder(self):
		"""
		decoder of VAE
		"""
		with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
			de_dense1 = tf.layers.dense(inputs = self.z_mean, units = self.inflate_to_size1, activation = None,
				kernel_initializer = self.init_w)
			de_dense1 = tf.layers.batch_normalization(de_dense1, training = self.is_training)
			de_dense1 = tf.nn.leaky_relu(de_dense1)
			de_dense1 = tf.layers.dropout(de_dense1, self.dropout_rate, training = self.is_training)            

			de_dense2 = tf.layers.dense(inputs=de_dense1, units = self.inflate_to_size2, activation=None,
				kernel_initializer = self.init_w)
			de_dense2 = tf.layers.batch_normalization(de_dense2, training = self.is_training)
			de_dense2 = tf.nn.leaky_relu(de_dense2)
			de_dense2 = tf.layers.dropout(de_dense2, self.dropout_rate, training = self.is_training)

			de_loc = tf.layers.dense(inputs=de_dense2, units= self.x_dim, activation=None, kernel_initializer = self.init_w)
			de_scale = tf.ones_like(de_loc)            
			return de_loc, de_scale

	def sample_posterior_z(self):
		"""
		sample the posterior latent samples for representation
		"""
		batch_size = tf.shape(self.mu)[0]
		eps = tf.random_normal(shape = [batch_size, self.z_dim])
		return self.mu + self.std * eps

	def sample_x(self):
		"""
		sample the reconstructed data
		"""
		batch_size_x = tf.shape(self.mu_x)[0]
		eps_x = tf.random_normal(shape = [batch_size_x, self.x_dim])
		return self.mu_x + self.std_x * eps_x

	def sample_z(self, batch_size, z_dim):
		"""
		sample the standard normal noises
		"""
		return np.random.normal(0.0, scale = 1.0, size = (batch_size, z_dim))
	
	def sample_data(self, data, batch_size):
		"""
		sample data from AnnData datatype
		"""
		lower = np.random.randint(0, data.shape[0] - batch_size)
		upper = lower + batch_size
		if sparse.issparse(data.X):
			x_mb = data[lower:upper, :].X.A
		else:
			x_mb = data[lower:upper, :].X
		return x_mb

	def sample_data_np(self, data, batch_size):
		"""
		sample data from numpy array datatype
		"""
		
		lower = np.random.randint(0, data.shape[0] - batch_size)
		upper = lower + batch_size
		
		return data[lower:upper]
	
	def log_prob_z_prior_dist(self):
		"""
		tensorflow prior distribution of latent variables
		"""
		batch_size = tf.shape(self.mu)[0]
		shape = [batch_size, self.z_dim]
		return ds.Normal(tf.zeros(shape), tf.ones(shape))

	def log_prob_z_prior(self):
		"""
		log probabilities of posterior latent samples on the prior 
		distribution
		"""
		return self.log_prob_z_prior_dist().log_prob(self.z_mean)

	def log_prob_x_dist(self):
		"""
		tensorflow normal distribution of the reconstructed data
		"""
		return ds.Normal(self.mu_x, self.std_x)

	def log_prob_z_post(self):
		"""
		log probabilities of posterior latent samples from their posterior distributions
		"""
		z_norm = (self.z_mean - self.mu) / self.std
		z_var = tf.square(self.std)
		return -0.5 * (z_norm * z_norm + tf.log(z_var) + np.log(2*np.pi))

	def qz_mss_entropies(self):
		"""
		estimate the minibatch entropies of the q(Z) and q(Zj) using 
		Minibatch Stratified Sampling (MSS)
		"""
		dataset_size = tf.convert_to_tensor(self.num_cells_train)
		batch_size = tf.shape(self.z_mean)[0]
		# compute the weights
		output = tf.zeros((batch_size - 1, 1))
		output = tf.concat([tf.ones((1,1)), output], axis = 0)
		outpart_1 = tf.zeros((batch_size, 1))
		outpart_3 = tf.zeros((batch_size, batch_size - 2))
		output = tf.concat([outpart_1, output], axis = 1)
		part_4 = - tf.concat([output, outpart_3], axis = 1)/tf.to_float(dataset_size)

		part_1 = tf.ones((batch_size, batch_size))/tf.to_float(batch_size - 1)
		part_2 = tf.ones((batch_size, batch_size))
		part_2 = - tf.matrix_band_part(part_2, 1, 0)/tf.to_float(dataset_size)

		part_3 = tf.eye(batch_size) * (2/tf.to_float(dataset_size) - 1/tf.to_float(batch_size - 1))

		weights =  tf_log(part_1 + part_2 + part_3 + part_4)

		# the entropies
		function_to_map = lambda x: self.log_prob_z_vector_post(tf.reshape(x, [1, self.z_dim])) 
		logqz_i_m = tf.map_fn(function_to_map, self.z_mean, dtype = tf.float32)
		weights_expand =  tf.expand_dims(weights, 2)
		logqz_i_margin = logsumexp(logqz_i_m + weights_expand, dim = 1, keepdims = False)
		logqz_value = tf.reduce_sum(logqz_i_m, axis = 2, keepdims = False)
		logqz_v_joint = logsumexp(logqz_value + weights, dim = 1, keepdims = False)
		logqz_sum = logqz_v_joint
		logqz_i_sum = logqz_i_margin

		marginal_entropies = (- tf.reduce_mean(logqz_i_sum, axis = 0))
		joint_entropies = (- tf.reduce_mean(logqz_sum)) 

		return marginal_entropies, joint_entropies

	def qz_entropies(self):
		"""
		estimate the large sample entropies of the q(Z) and q(Zj)
		"""
		batch_size = tf.shape(self.mu)[0]
		weights = - tf_log(tf.to_float(batch_size))

		function_to_map = lambda x: self.log_prob_z_vector_post(tf.reshape(x, [1, self.z_dim])) 
		logqz_i_m = tf.map_fn(function_to_map, self.z_mean, dtype = tf.float32)
		logqz_i_margin = logsumexp(logqz_i_m + weights, dim = 1, keepdims = False)
		logqz_value = tf.reduce_sum(logqz_i_m, axis = 2, keepdims = False)
		logqz_v_joint = logsumexp(logqz_value + weights, dim = 1, keepdims = False)
		logqz_sum = logqz_v_joint
		logqz_i_sum = logqz_i_margin

		marginal_entropies = (- tf.reduce_mean(logqz_i_sum, axis = 0))
		joint_entropies = (- tf.reduce_mean(logqz_sum)) 

		return marginal_entropies, joint_entropies


	def create_network(self):
		"""
		construct the VAE networks
		"""
		self.mu, self.std = self.encoder()
		self.z_mean = self.sample_posterior_z()
		self.z_mss_marginal_entropy, self.z_mss_joint_entropy = self.qz_mss_entropies()
		self.z_marginal_entropy, self.z_joint_entropy = self.qz_entropies()
		self.z_tc = total_correlation(self.z_marginal_entropy, self.z_joint_entropy)
		self.z_mss_tc = total_correlation(self.z_mss_marginal_entropy, self.z_mss_joint_entropy)
		self.mu_x, self.std_x = self.decoder()
		self.x_hat = self.sample_x()

	def loss_function(self):
		"""
		loss function of VAEs
		"""
		# KL divergence
		z_posterior = self.log_prob_z_post()
		z_prior = self.log_prob_z_prior()
		z_prior_sample = tf.reduce_sum(z_prior, [1])
		z_post_sample = tf.reduce_sum(z_posterior, [1])
		self.kl_loss = - tf.reduce_mean(z_prior_sample) + tf.reduce_mean(z_post_sample)
		
		# reconstruction error
		log_prob_x = self.log_prob_x_dist().log_prob(self.x)
		log_prob_x_sample = tf.reduce_sum(log_prob_x, [1])
		self.rec_x_loss = - tf.reduce_mean(log_prob_x_sample)

		# variables 
		tf_vars_all = tf.trainable_variables()
		evars  = [var for var in tf_vars_all if var.name.startswith("encoder")]
		dvars  = [var for var in tf_vars_all if var.name.startswith("decoder")]

		self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in evars + dvars])


		# Total correlation is self.z_mss_tc
		self.tcvae_loss = self.alpha * self.kl_loss + self.rec_x_loss + self.beta * self.z_mss_tc
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.solver = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.tcvae_loss, 
				var_list = evars + dvars)

	def log_prob_z_vector_post(self, z_vector):
		"""
		log probabilities of latent variables on given posterior normal distributions
		"""
		z_norm = (z_vector - self.mu) / self.std
		z_var = tf.square(self.std)
		return -0.5 * (z_norm * z_norm + tf.log(z_var) + np.log(2*np.pi))
	
	def encode(self, x_data):
		"""
		encode data to the latent samples
		"""
		latent = self.sess.run(self.z_mean, feed_dict = {self.x: x_data, self.is_training: False})
		return latent

	def encode_mean(self, x_data):
		"""
		encode data to the latent means
		"""
		latent = self.sess.run(self.mu, feed_dict = {self.x: x_data, self.is_training: False})
		return latent

	def decode(self, z):
		"""
		decode to data fromm latent values
		"""
		data = self.sess.run(self.x_hat, feed_dict = {self.z_mean: z, self.is_training: False})
		return data
	
	
	def avg_vector(self, data):
		"""
		encode data to the latent sample means
		"""
		latent = self.encode(data)
		latent_avg = np.average(latent, axis = 0)
		return latent_model_parameter

	@property
	def model_parameter(self):
		"""
		report the number of training parameters
		"""
		self.total_param = self.sess.run(self.parameter_count)
		return "There are {} parameters in VAE.".format(self.total_param)

	

	def reconstruct(self, data, if_latent = False):
		"""
		reconstruct data from original data or latent samples
		"""
		if if_latent:
			latent = data
		else:
			latent = self.encode(data)

		rec_data = self.sess.run(self.x_hat, feed_dict = {self.z_mean: latent, self.is_training: False})
		return rec_data

	def restore_model(self, model_path):
		"""
		restore model from model_path
		"""
		self.saver.restore(self.sess, model_path)

	def save_model(self, model_save_path, epoch):
		"""
		save the trained model to the model_save_path
		"""
		os.makedirs(model_save_path, exist_ok = True)
		model_save_name = os.path.join(model_save_path, "model")
		save_path = self.saver.save(self.sess, model_save_name, global_step = epoch)

		np.save(os.path.join(model_save_path, "training_time.npy"), self.training_time)
		np.save(os.path.join(model_save_path, "train_loss.npy"), self.train_loss)
		np.save(os.path.join(model_save_path, "valid_loss.npy"), self.valid_loss)

	
	def train_np(self, train_data, use_validation = False, valid_data = None, use_test_during_train = False, test_data = None,
				 test_every_n_epochs = 100, test_size = 3000, inception_score_data = None, n_epochs = 25, batch_size = 128, early_stop_limit =20, 
				 threshold = 0.0025, shuffle = True, save = False, model_save_path = None, output_save_path = None, verbose = False):
		"""
		train VAE with train_data (numpy array) and optional valid_data (numpy array) for n_epochs.
		"""
		log.info("--- Training ---")
		if use_validation and valid_data is None:
			raise Exception("valid_data is None but use_validation is True.")

		patience = early_stop_limit
		min_delta = threshold
		patience_cnt = 0

		n_train = train_data.shape[0]
		n_valid = None
		if use_validation:
			n_valid = valid_data.shape[0]

		# generation performance at the PC space
		if use_test_during_train:
			pca_data_50 = PCA(n_components = 50, random_state = 42)
			genmetric = MetricVisualize()
			RFE = RandomForestError()
			genmetrics_pd = pd.DataFrame({'epoch':[], 'is_real_mu': [], 'is_real_std': [], 
										  'is_fake_mu':[], 'is_fake_std':[], 'rf_error':[]})

			pca_data_fit = pca_data_50.fit(train_data)
		
		if shuffle:
			index_shuffle = list(range(n_train))

		for epoch in tqdm(range(1, n_epochs + 1)):

			begin = time.time()

			if shuffle:
				np.random.shuffle(index_shuffle)
				train_data = train_data[index_shuffle]

				if inception_score_data is not None:
					inception_score_data = inception_score_data[index_shuffle]

			train_loss, valid_loss = 0.0, 0.0

			for _ in range(1, n_train // batch_size + 1):
				x_mb = self.sample_data_np(train_data, batch_size)
				z_mb = self.sample_z(batch_size, self.z_dim)
				_, current_loss_train = self.sess.run([self.solver, self.tcvae_loss], 
													   feed_dict = {self.x: x_mb, self.z: z_mb, 
																	self.is_training: self.if_BNTrainingMode})

				train_loss += current_loss_train * batch_size

			train_loss /= n_train
			
			if use_validation:
				valid_loss = 0
				for _ in range(1, n_valid // batch_size + 1):
					x_mb = self.sample_data_np(valid_data, batch_size)
					z_mb = self.sample_z(batch_size, self.z_dim)
					current_loss_valid = self.sess.run(self.tcvae_loss, 
													   feed_dict = {self.x: x_mb, self.z: z_mb, 
																	self.is_training: False})
					valid_loss += current_loss_valid * batch_size
				
				valid_loss /= n_valid

			self.train_loss.append(train_loss)
			self.valid_loss.append(valid_loss)
			self.training_time += (time.time() - begin)

			# testing for generation metrics
			if (epoch - 1) % test_every_n_epochs == 0 and use_test_during_train:
				
				if test_data is None:
					reset_test_data = True
					sampled_indices = sample(range(n_train), test_size)
					
					test_data = train_data[sampled_indices, :]
					gen_data = self.reconstruct(test_data)

					if inception_score_data is not None:
						inception_score_subdata = inception_score_data[sampled_indices]
						mean_is_real, std_is_real = genmetric.InceptionScore(test_data, inception_score_subdata, test_data)
						mean_is_fake, std_is_fake = genmetric.InceptionScore(test_data, inception_score_subdata, gen_data)
					else:
						mean_is_real = std_is_real = mean_is_fake = std_is_fake = 0.0

				else:
					assert test_data.shape[0] == test_size
					reset_test_data = False

					gen_data = self.reconstruct(test_data)

					if inception_score_data is not None:
						inception_score_subdata = inception_score_data
						mean_is_real, std_is_real = genmetric.InceptionScore(test_data, inception_score_subdata, test_data)
						mean_is_fake, std_is_fake = genmetric.InceptionScore(test_data, inception_score_subdata, gen_data)
					else:
						mean_is_real = std_is_real = mean_is_fake = std_is_fake = 0.0


				errors_d = list(RFE.fit(test_data, gen_data, pca_data_fit, if_dataPC = True, output_AUC = False)['avg'])[0]
				genmetrics_pd = pd.concat([genmetrics_pd, pd.DataFrame([[epoch, mean_is_real, std_is_real, mean_is_fake, std_is_fake, 
						errors_d]], columns = ['epoch', 'is_real_mu', 'is_real_std', 'is_fake_mu', 'is_fake_std', 'rf_error'])])
				
				if save:
					genmetrics_pd.to_csv(os.path.join(output_save_path, "GenerationMetrics.csv"))
				if reset_test_data:
					test_data = None


			if verbose: 
				print(f"Epoch {epoch}: Train Loss: {train_loss} Valid Loss: {valid_loss}")

			# early stopping
			if use_validation and epoch > 1:
				if self.valid_loss[epoch - 2] - self.valid_loss[epoch - 1] > min_delta:
					patience_cnt = 0
				else:
					patience_cnt += 1

				if patience_cnt > patience:
					if save:
						self.save_model(model_save_path, epoch)
						log.info(f"Model saved in file: {model_save_path}. Training stopped earlier at epoch: {epoch}.")
						if verbose:
							print(f"Model saved in file: {model_save_path}. Training stopped earlier at epoch: {epoch}.")
						if use_test_during_train:
							genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))
					break


		if save:
			self.save_model(model_save_path, epoch)
			log.info(f"Model saved in file: {model_save_path}. Training finished.")
			if verbose:
				print(f"Model saved in file: {model_save_path}. Training finished.")
			if use_test_during_train:
				genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))

	def train_np_crossValidate(self, train_data, seed = 123, use_test_during_train = False,
							   test_every_n_epochs = 100, test_size = 3000, n_epochs = 25, batch_size = 128, early_stop_limit =20,
							   threshold = 0.0025, shuffle = True, save = False, model_save_path = None,
							   output_save_path = None, verbose = False,cv_prop = 0.8):
		"""
		train VAE with train_data (numpy array) and optional valid_data (numpy array) for n_epochs.
		force to use validation
		"""
		log.info("--- Training ---")

		patience = early_stop_limit
		min_delta = threshold
		patience_cnt = 0

		n_data = train_data.shape[0]
		n_train = int(n_data * cv_prop)
		n_valid = n_data - n_train
		n_test = n_valid

		# a random split of train and validation datasets
		random_state = np.random.RandomState(seed = seed)
		permutation = random_state.permutation(n_data)
		indices_test, indices_train = permutation[:n_test], permutation[n_test:]

		train_data_train = train_data[indices_train]
		train_data_test = train_data[indices_test]

		# generation performance at the PC space
		if use_test_during_train:
			pca_data_50 = PCA(n_components = 50, random_state = 42)
			genmetric = MetricVisualize()
			RFE = RandomForestError()
			genmetrics_pd = pd.DataFrame({'epoch':[], 'rf_train':[], 'rf_test':[]})
			pca_data_fit = pca_data_50.fit(train_data)

		if shuffle:
			index_shuffle_train = list(range(n_train))
			index_shuffle_test = list(range(n_test))

		for epoch in range(1, n_epochs + 1):

			begin = time.time()
			if shuffle:
				np.random.shuffle(index_shuffle_train)
				train_data_train = train_data_train[index_shuffle_train]

				np.random.shuffle(index_shuffle_test)
				train_data_test = train_data_test[index_shuffle_test]

			train_loss, valid_loss = 0.0, 0.0

			for _ in range(1, n_train // batch_size + 1):
				x_mb = self.sample_data_np(train_data_train, batch_size)
				z_mb = self.sample_z(batch_size, self.z_dim)
				_, current_loss_train = self.sess.run([self.solver, self.tcvae_loss],
													   feed_dict = {self.x: x_mb, self.z: z_mb,
																	self.is_training: self.if_BNTrainingMode})

				train_loss += current_loss_train * batch_size

			train_loss /= (batch_size * (n_train // batch_size + 1))

			for _ in range(1, n_test // batch_size + 1):
				x_mb = self.sample_data_np(train_data_test, batch_size)
				z_mb = self.sample_z(batch_size, self.z_dim)
				current_loss_valid = self.sess.run(self.tcvae_loss,
												   feed_dict = {self.x: x_mb, self.z: z_mb,
																self.is_training: False})
				valid_loss += current_loss_valid * batch_size

			valid_loss /= (batch_size * (n_test // batch_size + 1))

			self.train_loss.append(train_loss)
			self.valid_loss.append(valid_loss)
			self.training_time += (time.time() - begin)

			# testing for generation metrics
			if (epoch - 1) % test_every_n_epochs == 0 and use_test_during_train:

				sampled_indices_train = sample(range(n_train), test_size)
				test_data_train = train_data_train[sampled_indices_train, :]
				gen_data_train = self.reconstruct(test_data_train)

				sampled_indices_test = sample(range(n_test), test_size)
				test_data_test = train_data_test[sampled_indices_test, :]
				gen_data_test = self.reconstruct(test_data_test)

				errors_train = list(RFE.fit(test_data_train, gen_data_train, pca_data_fit, if_dataPC = True, output_AUC = False)['avg'])[0]
				errors_test = list(RFE.fit(test_data_test, gen_data_test, pca_data_fit, if_dataPC = True, output_AUC = False)['avg'])[0]
				genmetrics_pd = pd.concat([genmetrics_pd, pd.DataFrame([[epoch, errors_train, errors_test]],
										   columns = ['epoch', 'rf_train', 'rf_test'])])
				if save:
					genmetrics_pd.to_csv(os.path.join(output_save_path, "GenerationMetrics.csv"))


			if verbose:
				print(f"Epoch {epoch}: Train Loss: {train_loss} Valid Loss: {valid_loss}")


		if save:
			self.save_model(model_save_path, epoch)
			log.info(f"Model saved in file: {model_save_path}. Training finished.")
			if verbose:
				print(f"Model saved in file: {model_save_path}. Training finished.")
			if use_test_during_train:
				genmetrics_pd.to_csv(os.path.join(model_save_path, "GenerationMetrics.csv"))