import os
import cv2
import math
import numpy as np

class DataLoader(object):
	def __init__(self, data_dir):
		self.train_dir = os.path.join(data_dir, 'train')
		self.test_dir = os.path.join(data_dir, 'test')

		self.dir = {}
		self.dir[1] = self.train_dir
		self.dir[0] = self.test_dir

		assert os.path.exists(self.train_dir), "train dir doesn't exists"
		assert os.path.exists(self.test_dir), "test dir doesn't exists"


	def load_data(self, train):
		face_dir = os.path.join(self.dir[train], 'face')
		non_face_dir = os.path.join(self.dir[train], 'non-face')

		faces_list = os.listdir(face_dir)
		non_faces_list = os.listdir(non_face_dir)

		faces = []

		for f in faces_list:
			img_path = os.path.join(face_dir, f)
			_img = self.load_image(img_path)
			if (_img is not None):
				faces.append(_img)

		non_faces = []
		for f in non_faces_list:
			img_path = os.path.join(non_face_dir, f)
			_img = self.load_image(img_path)
			if (_img is not None):
				non_faces.append(_img)

		return faces, non_faces

	def flatten_images(self, images):
		return [self.flatten_image(img) for img in images]

	def unflatten_images(self, images):
		return [self.unflatten_image(img) for img in images]

	def flatten_image(self, img):
		dim = img.shape
		return img.reshape(dim[0]*dim[1]*dim[2])

	def unflatten_image(self, img):
		dim3 = 3
		temp = img.shape[0]/3
		dim1 = int(math.sqrt(temp))
		return img.reshape((dim1, dim1, dim3))

	def mean_face(self, faces):
		m = np.mean(faces, axis=0)
		return self.unflatten_image(m)


	def load_image(self, img_path):
		try:
			_img = cv2.imread(img_path)
			return _img
		except:
			print "Unable to laod image from ", img_path
			return None
