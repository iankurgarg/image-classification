import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal

from BaseEM import BaseEM

class MixtureOfGaussian(BaseEM):
	def __init__(self, n_components=2):
		self.n_components = n_components
		self.params = {}
		self.weights = None
		self.prior_probs = None
		self.models = {}

		# np.random.seed(1)

	def calculate_prior_probs(self):
		# self.weights should be of shape (n_samples, n_components)
		prior_probs = np.sum(self.weights, axis=0)
		N = np.sum(prior_probs)

		prior_probs = prior_probs / N
		return prior_probs

	def maximization(self, data):
		n_samples, n_features = data.shape
		# print self.weights[:,0]
		for i in range(self.n_components):
			self.params[i]['mean'] = np.average(data, axis=0, weights=self.weights[:,i], returned=False)
			self.params[i]['cov'] = np.cov(data, rowvar=False, aweights=self.weights[:,i])

		self.prior_probs = self.calculate_prior_probs()
		# print "self.prior_probs = ", self.prior_probs
		self.update_model()


	def expectation(self, data):
		weights = []
		for i in range(self.n_components):
			wi = self.models[i].pdf(data)
			weights.append(wi)
		
		weights = np.matrix(weights).transpose()

		# Element wise multiply with corresponding prior probs
		weights = np.multiply(weights, self.prior_probs)
		# normalize by dividing with sum of posteriors. weights here should be of shape = n_samples, n_components
		self.weights = normalize(weights, norm='l1', axis=1)
		# print "self.weights = ", self.weights

	def update_model(self):
		for i in range(self.n_components):
			self.models[i] = multivariate_normal(mean=np.array(self.params[i]['mean']).ravel(), cov=self.params[i]['cov'])
	
	# Function for initial setup for EM
	def setup(self, data):
		n_samples, n_features = data.shape

		self.weights = []
		for i in range(self.n_components):
			a = [1/float(self.n_components)]*n_samples
			self.weights.append(a)

		self.weights = np.matrix(self.weights).transpose()

		for i in range(self.n_components):
			self.params[i] = {}
			l = int(np.random.rand()*data.shape[0])
			self.params[i]['mean'] = data[l]
			# A = np.random.rand(n_features, n_features)
			# self.params[i]['cov'] = np.dot(A,A.transpose())
			self.params[i]['cov'] = np.cov(data, rowvar=False)


		# shape of prior probs here should be (1, n_components)
		self.prior_probs = self.calculate_prior_probs()
		self.update_model()
		self.L = self.calcualte_overall_likelihood(data)

	def pdf(self, X):
		likelihood_probs = []

		for i in range(self.n_components):
			w = self.models[i].pdf(X)
			likelihood_probs.append(w)

		likelihood_probs = np.matrix(likelihood_probs).transpose()

		posterior_probs = np.multiply(likelihood_probs, self.prior_probs)

		# likelihood_probs should be of shape (n_samples, 1)
		posterior_probs = np.sum(posterior_probs, axis=1)

		return posterior_probs
