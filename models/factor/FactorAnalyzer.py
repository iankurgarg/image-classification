import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import multivariate_normal
import numpy as np

from sklearn.preprocessing import normalize


class FactorAnalyzer(object):
	def __init__(self, n_components=2):
		self.n_components = n_components
		
		self.mean, self.cov, self.phi = None, None, None

		self.Eh, self.Ehh = None, None
		
		self.max_iter = 400
		self.D = 1.0
		self.Identity = None
		# np.random.seed(1)

	def maximization(self, data):
		s = self.phi.shape
		I = len(data)
		t1 = np.matrix(np.zeros((s[0], s[1])))
		for i in range(I):
			t1 += data[i,:].transpose()*self.Eh[:,i].transpose()

		t2 = np.linalg.inv(np.sum(self.Ehh, axis=0))
		new_phi = t1*t2
		assert self.phi.shape == new_phi.shape
		self.phi = new_phi


		new_cov = np.matrix(np.zeros((s[0],s[0])))
		for i in range(I):
			t1 = data[i,:].transpose()*data[i,:]
			t2 = self.phi*self.Eh[:,i]*data[i,:]

			new_cov += (t1-t2)

		new_cov /= I
		assert new_cov.shape == self.cov.shape
		for i in range(len(self.cov)):
			self.cov[i,i] = new_cov[i,i]

		self.update_model()

	def expectation(self, data):
		cov_inv = np.linalg.inv(self.cov)
		
		t1 = np.linalg.inv((self.phi.transpose()*cov_inv*self.phi) + self.Identity)
		
		for i in range(len(data)):
			self.Eh[:,i] = t1*self.phi.transpose()*cov_inv*(data[i,:] - self.mean).transpose()
			self.Ehh[i] = t1  + (self.Eh[:,i] * self.Eh[:,i].transpose())


	def calcualte_overall_likelihood(self, data):
		w = self.pdf(data)
		return np.sum(np.log(w))

	def update_model(self):
		self.actual_cov = self.phi*self.phi.transpose() + self.cov
		self.model = multivariate_normal(mean=np.array(self.mean).ravel(), cov=self.actual_cov)
	
	# checks if termination condition is achieved
	def terimation(self, data):
		new_L = self.calcualte_overall_likelihood(data)
		if (self.L is None):
			self.L = new_L
			return False
		
		diff = new_L - self.L
		# print "new_L = ", new_L, " L = ", self.L
		# print "percentage change = ", diff/float(abs(self.L))
		if (diff > 0 and diff/float(abs(self.L)) > 0.00001) :
			self.L = new_L
			return False
		
		return True

	def run(self, data, K):
		i = 0
		data = np.matrix(data, dtype=float)
		self.setup(data, K)

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
	def setup(self, data, K):
		n_samples, self.D = data.shape

		self.mean = np.mean(data, axis=0)
		self.cov = np.matrix(np.zeros((self.D, self.D), dtype=float))

		for i in range(self.D):
			self.cov[i,i] = np.var(data[:,i])

		self.phi = np.matrix(np.random.rand(self.D, K))
		self.Identity = np.identity(K)

		self.Eh = np.matrix(np.zeros((K, n_samples)))
		self.Ehh = np.zeros((n_samples, K, K))

		# print "Setup Complete: "
		# print "Mean = ", self.mean.shape
		# print "Cov = ", self.cov.shape
		# print "Eh = ", self.Eh.shape
		# print "Ehh = ", self.Ehh.shape
		# print "Identity = ", self.Identity.shape
		# print "Phi = ", self.phi.shape

		self.update_model()
		self.L = self.calcualte_overall_likelihood(data)
		# print "L = ", self.L

	def pdf(self, X):
		return self.model.pdf(X)




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
