import os, sys
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
sys.path.append(path)

from scipy.stats import multivariate_normal
from data_io.DataLoader import DataLoader
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from BaseModel import BaseModel
from Visualization import *

from gaussian.Gaussian import Gaussian
from gaussian.MixtureOfGaussian import MixtureOfGaussian
from t_dist.TDistribution import TDistribution
from factor.FactorAnalyzer import FactorAnalyzer



class Model(BaseModel):
	def __init__(self, mtype='mog'):
		self.dist1 = None
		self.dist2 = None
		self.thresh = 0.5
		self.model_type = mtype

	def fit(self, face, non_face):
		trainX = np.concatenate((face, non_face))
		trainY = [1]*len(face) + [0]*len(non_face)

		pcaX = self.preprocess(trainX, trainY)

		pca_face = pcaX[:1000]
		pca_non_face = pcaX[1000:]

		if (self.model_type == 'g'):
			self.dist1 = Gaussian()
			self.dist1.run(pca_face)

			self.dist2 = Gaussian()
			self.dist2.run(pca_non_face)
		elif (self.model_type == 'mog'):
			self.dist1 = MixtureOfGaussian(4)
			self.dist1.run(pca_face)

			self.dist2 = MixtureOfGaussian(4)
			self.dist2.run(pca_non_face)
		elif (self.model_type == 't-dist'):
			self.dist1 = TDistribution()
			self.dist1.run(pca_face)

			self.dist2 = TDistribution()
			self.dist2.run(pca_non_face)
		elif (self.model_type == 'factor'):
			self.dist1 = FactorAnalyzer(50)
			self.dist1.run(pca_face)

			self.dist2 = FactorAnalyzer(50)
			self.dist2.run(pca_non_face)



	def preprocess(self, trainX, trainY):
		self.scaler_model = StandardScaler()
		scaledX = self.scaler_model.fit_transform(trainX, trainY)

		self.pca_model = PCA(n_components=10)
		pcaX = self.pca_model.fit_transform(scaledX)

		return pcaX

	def converted_mean(self):
		converted_means = []
		w = self.dist1.mean_()
		c = self.pca_model.components_
		
		for a in w:
			cmean = np.average(c, axis=0, weights=a)
			converted_means.append(cmean)

		c = int(np.sqrt(len(converted_means)))
		if (c*c < len(converted_means)):
			c += 1

		plt.figure(1, figsize=(10, 10))
		plt.title('Mean')
		for i, a in enumerate(converted_means):
			plt.subplot((100*c)+(10*c) + i + 1)
			plt.imshow(a.reshape((60,60)), cmap="gray")

		plt.show()
		return converted_means

	def converted_covariance(self):
		converted_vars = []
		w = self.dist1.variance_()
		c = np.matrix(self.pca_model.components_)
		for a in w:
			cmean = c.transpose()*a*c
			converted_vars.append(cmean)

		c = int(np.sqrt(len(converted_vars)))
		if (c*c < len(converted_vars)):
			c += 1

		plt.figure(1, figsize=(10, 10))
		plt.title('Covariance')
		for i, a in enumerate(converted_vars):
			plt.subplot((100*c)+(10*c) + i + 1)
			cov_diag = np.diag(a)
			plt.imshow(cov_diag.reshape((60,60)), cmap="gray")

		plt.show()

		return converted_vars

	def predict(self, img):
		if (len(img.shape) == 1):
			img = img.reshape(1, -1)
		
		scaled_img = self.scaler_model.transform(img)
		pca_img = self.pca_model.transform(scaled_img)
		a = self.dist1.pdf(pca_img)
		b = self.dist2.pdf(pca_img)
		
		probs = a / (a+b)

		labels = []
		for a in probs:
			if a > self.thresh:
				labels.append(1)
			else:
				labels.append(0)

		return labels



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Run all Models")
	parser.add_argument('-i', '--input', action="store", dest="input_dir", help="Directory containing input images <path>", required=True)
	parser.add_argument('-m', '--model', action="store", dest="model", help="Model to be run", choices=['g', 'mog', 't-dist', 'factor'], required=True)

	options = parser.parse_args(sys.argv[1:])

	dl = DataLoader(options.input_dir, rgb=0)

	face, non_face = dl.load_data(train=1)

	m = Model(mtype=options.model)
	m.fit(face, non_face)

	means = m.converted_mean()
	covars = m.converted_covariance()

	test_face, test_non_face = dl.load_data(train=0)
	testX = np.concatenate((test_face, test_non_face))
	testY = [1]*len(test_face) + [0]*len(test_non_face)


	predicted = m.predict(testX)

	print "false_positive_rate = ", m.false_positive_rate(testY, predicted)
	print "false_negative_rate = ", m.false_negative_rate(testY, predicted)
	print "misclassification_rate = ", m.misclassification_rate(testY, predicted)