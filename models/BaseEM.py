import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import multivariate_normal
import numpy as np

from sklearn.preprocessing import normalize


class BaseEM(object):
	n_components = 2
	max_iter = 400
	min_percentage_change = 0.00001
	L = 0.0

	def maximization(self, data):
		pass

	def expectation(self, data):
		pass


	def calcualte_overall_likelihood(self, data):
		w = self.pdf(data)
		return np.sum(np.log(w))

	def update_model(self):
		pass
	
	# checks if termination condition is achieved
	def terimation(self, data):
		new_L = self.calcualte_overall_likelihood(data)
		if (self.L is None):
			self.L = new_L
			return False
		
		diff = new_L - self.L
		
		if (diff > 0 and diff/float(abs(self.L)) > self.min_percentage_change):
			self.L = new_L
			return False
		
		return True

	def run(self, data):
		i = 0
		data = np.matrix(data, dtype=float)
		self.setup(data)

		while (i < self.max_iter):
			i += 1
			# print "data.shape = ", data.shape
			self.expectation(data)
			self.maximization(data)

			if self.terimation(data):
				break;

			# print "Completed Iteration ", i

		print "Finished EM. Last Iteration Number = ", i, " max iter = ", self.max_iter

	
	# Function for initial setup for EM
	def setup(self, data):
		pass

	def pdf(self, X):
		pass
