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
		mean = np.mean(data, axis=0)
		cov = np.cov(data, rowvar=False)

		self.model = multivariate_normal(mean=mean, cov=cov)

	def pdf(self, X):
		return self.model.pdf(X)
