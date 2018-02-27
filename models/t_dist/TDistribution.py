import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import multivariate_normal
from scipy.special import digamma, gammaln, gamma
from data_io.DataLoader import DataLoader
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from Visualization import *


class TDistribution(object):
	def __init__(self, n_components=2):
		self.n_components = n_components
		self.max_v = 100
		self.mean, self.cov, self.v = None, None, float(self.max_v)
		
		self.delta, self.Eh, self.Elogh = None, None, None
		
		self.max_iter = 40
		self.models = {}

		self.L = None
		self.D = 1.0

		self.tcosts = []

		# np.random.seed(1)

	# def gamma(self, x):
	# 	return gamma.pdf(x, a=self.v/2, scale=self.v/2)


	def tCost(self, v):
		v = float(v)
		t1 = (v/2)*np.log(v/2)
		t2 = (gammaln(v/2))

		cost = 0.0

		for i in range(self.Elogh.shape[0]):
			t3 = -((v/2) - 1)*self.Elogh[i,0]
			t4 = (v/2)*self.Eh[i,0]
			cost += (t1 + t2 + t3 + t4)

		return -cost

	def maximization(self, data):
		temp = np.multiply(data, self.Eh)
		# N = np.sum(self.Eh)
		# print "N = ", N
		self.mean = np.mean(temp, axis=0)/np.sum(self.Eh)

		self.cov = np.zeros((data.shape[1], data.shape[1]), dtype=float)
		for i in range(data.shape[0]):
			assert data[i,:].shape == self.mean.shape, str(i) + " data.shape = " + str(data.shape) + ":" + str(data[i, :].shape) + " != " + str(self.mean.shape)
			temp = np.subtract(data[i,:], self.mean)
			temp2 = temp.transpose() * temp
			assert temp2.shape == self.cov.shape
			self.cov += (self.Eh[i,0]*temp2)

		self.cov /= np.sum(self.Eh)
		
		for i in range(1, self.max_v+1):
			self.tcosts[i-1] = self.tCost(i)

		# print "self.tcosts = ", self.tcosts
		# print "np.argmin(self.tcosts) = ", np.argmin(self.tcosts)
		self.v = float(np.argmin(self.tcosts) + 1)


	def expectation(self, data):
		
		# print "self.delta.shape = ", self.delta.shape
		# print "len(self.delta) = ", len(self.delta)
		cov_inv = np.linalg.inv(self.cov)
		# print "cov_inv.shape = ", cov_inv.shape
		for i in range(len(self.delta)):
			# print "data[i,:].shape = ", data[i,:].shape
			# print "self.mean.shape = ", self.mean.shape
			assert data[i,:].shape == self.mean.shape, str(i) + " data.shape = " + str(data.shape) + ":" + str(data[i, :].shape) + " != " + str(self.mean.shape)
			temp = np.subtract(data[i,:], self.mean)
			temp2 = (temp * cov_inv) * temp.transpose()
			self.delta[i,0] = temp2[0,0]

		# print "sum(delta) = ", np.sum(self.delta)
		# print "self.v = ", self.v
		# print "np.reciprocal(self.delta + self.v) = ", np.sum(np.reciprocal(self.delta + self.v, dtype=float))
		# print "self.v + self.D = ", self.v + self.D
		self.Eh = np.reciprocal(self.delta + self.v, dtype=float) * (self.v + self.D)
		self.Elogh = digamma(self.v/2 + self.D/2) - np.log(self.v/2 + self.delta/2, dtype=float)



	def calcualte_overall_likelihood(self, data):
		for i in range(len(self.delta)):
			temp = np.subtract(data[i,:], self.mean)
			temp2 = (temp * np.linalg.inv(self.cov)) * temp.transpose()
			self.delta[i,0] = temp2[0,0]
		
		I = data.shape[0]

		# print "(self.v+self.D)/2) = ", (self.v+self.D)/2
		t1 = I*(gammaln(float(self.v+self.D)/2))
		t2 = I*self.D*np.log(self.v*np.pi)/2
		t3 = I*np.log(np.linalg.det(self.cov))/2
		t4 = I*(gammaln(self.v/2))
		t5 = 0.0

		for i in range(self.Eh.shape[0]):
			t5 +=  np.log(1 + self.delta[i,0]/self.v)/2

		t5 = (self.v + self.D)*t5

		L = t1-t2-t3-t4-t5
		
		return L
	
	# checks if termination condition is achieved
	def terimation(self, data):
		new_L = self.calcualte_overall_likelihood(data)
		if (self.L is None):
			self.L = new_L
			return False
		
		diff = new_L - self.L
		
		# print "percentage change = ", diff/float(abs(self.L))
		if (diff > 0.001):
			self.L = new_L
			return False
		
		return True

	def run(self, data):
		i = 0
		data = np.matrix(data, dtype=float)
		self.setup(data)
		# print "self.weights = ", self.weights
		# print "self.prior_probs = ", self.prior_probs
		
		while (i < self.max_iter):
			i += 1
			# print "data.shape = ", data.shape
			self.expectation(data)
			self.maximization(data)

			if self.terimation(data):
				break;

			# print "Completed Iteration ", i

		print "Finished EM. Last Iteration Number = ", i, " max iter = ", self.max_iter
		# print "self.v = ", self.v
		# print "self.mean = ", self.mean
		# print "self.cov = ", self.cov

		# for i in range(self.n_components):
		# 	m = multivariate_normal(mean=self.params[i]['mean'], cov=self.params[i]['cov'])
		# 	self.models[i] = m
	
	# Function for initial setup for EM
	def setup(self, data):
		n_samples, self.D = data.shape
		self.D = float(self.D)
		self.delta = np.matrix([1]*n_samples, dtype=float).transpose()

		self.mean = np.mean(data, axis=0)
		# print "data type = ", type(data)
		# print "self.mean.shape = ", self.mean.shape
		self.cov = np.cov(data, rowvar=False)
		self.v = 1000
		self.tcosts = [0]*self.max_v

		# self.L = self.calcualte_overall_likelihood(data)

	def pdf(self, X):
		t1_num = gamma((self.v+self.D)/2)
		t1_denom = pow(self.v*np.pi, self.D/2) * np.sqrt(np.linalg.det(self.cov)) * gamma(self.v/2)

		temp = np.subtract(X, self.mean)
		t2_num = (temp * np.linalg.inv(self.cov)) * temp.transpose()
		t2_num = np.diag(t2_num)

		t2 =  1 + (t2_num)/self.v
		t2 = np.float_power(t2, -(self.v + self.D)/2)

		posterior_probs = t2 * (t1_num/t1_denom)

		return posterior_probs



if __name__ == '__main__':
	dl = DataLoader("/Users/iankurgarg/Code/Vision/Project-1/image-classification/images-2")

	face, non_face = dl.load_data(train=1)

	# show_mean(face, dim3=3)
	# show_cov(face, dim3=3)

	m = Model3()
	m.run(face)

	# test_face, test_non_face = dl.load_data(train=0)
	# testX = np.concatenate((test_face, test_non_face))
	# testY = [1]*len(test_face) + [0]*len(test_non_face)
