import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal


class Gaussian(object):
	def __init__(self):
		self.thresh = 0.5
		self.model = None

	def run(self, data):
		self.mean = np.mean(data, axis=0)
		self.cov = np.cov(data, rowvar=False)

		self.model = multivariate_normal(mean=self.mean, cov=self.cov)

	def pdf(self, X):
		return self.model.pdf(X)

	def mean_(self):
		return [self.mean]

	def variance_(self):
		return [self.cov]
