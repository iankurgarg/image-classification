import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append('../')

from scipy.stats import multivariate_normal
from data_io.DataLoader import DataLoader
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from BaseModel import BaseModel
from Visualization import *

from EM import EM



class Model3(BaseModel):
	def __init__(self):
		self.dist1 = None
		self.dist2 = None
		self.thresh = 0.5

	def fit(self, face, non_face):
		trainX = np.concatenate((face, non_face))
		trainY = [1]*len(face) + [0]*len(non_face)

		pcaX = self.preprocess(trainX, trainY)

		pca_face = pcaX[:1000]
		pca_non_face = pcaX[1000:]

		self.dist1 = EM()
		self.dist1.run(pca_face)

		self.dist2 = EM()
		self.dist2.run(pca_non_face)

	def preprocess(self, trainX, trainY):
		self.scaler_model = StandardScaler()
		scaledX = self.scaler_model.fit_transform(trainX, trainY)

		self.pca_model = PCA(n_components=30)
		pcaX = self.pca_model.fit_transform(scaledX)

		return pcaX

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
	dl = DataLoader("/Users/iankurgarg/Code/Vision/Project-1/image-classification/images-2")

	face, non_face = dl.load_data(train=1)

	m = Model3()
	m.fit(face, non_face)

	test_face, test_non_face = dl.load_data(train=0)
	testX = np.concatenate((test_face, test_non_face))
	testY = [1]*len(test_face) + [0]*len(test_non_face)


	# predicted = m.predict(testX)

	# print "false_positive_rate = ", m.false_positive_rate(testY, predicted)
	# print "false_negative_rate = ", m.false_negative_rate(testY, predicted)
	# print "misclassification_rate = ", m.misclassification_rate(testY, predicted)