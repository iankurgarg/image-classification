import os

import cv2
import math, random

import urllib
import numpy as np
import pandas as pd


class DataPreparation(object):
	def __init__(self, input_dir, output):
		self.input_dir = input_dir
		self.output_dir = output

		if (not os.path.exists(self.output_dir)):
			os.makedirs(self.output_dir)

		self.out_dir = {}
		self.out_dir[1] = os.path.join(self.output_dir, 'train')
		self.out_dir[0] = os.path.join(self.output_dir, 'test')

		if (not os.path.exists(self.out_dir[1])):
			os.makedirs(self.out_dir[1])

		if (not os.path.exists(self.out_dir[0])):
			os.makedirs(self.out_dir[0])

		annotation_file_path = os.path.join(self.input_dir, 'umdfaces_batch3_ultraface.csv')

		self.annotations = pd.read_csv(annotation_file_path, encoding="utf-8-sig")
		self.annotations.index = self.annotations.FILE

		self.img_dim = (60, 60)

		self.debug_mode = False
		self.debug_count = 0
		random.seed(1)

	def run(self, train_images, test_images):
		persons = os.listdir(self.input_dir)
		persons = [x for x in persons if os.path.isdir(os.path.join(self.input_dir, x))]
		random.shuffle(persons)
		N = len(persons)
		tN = int(0.8*N)
		train_persons = persons[:tN]
		test_persons = persons[tN:]

		print "Loading Face Training Images"
		face_imgs = self.generate_face_images(train_persons, train_images)
		self.save_images(face_imgs, os.path.join(self.out_dir[1], 'face'))

		print "Loading Face Test Images"
		face_imgs = self.generate_face_images(test_persons, test_images)
		self.save_images(face_imgs, os.path.join(self.out_dir[0], 'face'))

		print "Loading Background Images"
		bg_images = self.generate_background_images(train_images+test_images)
		self.save_images(bg_images[:train_images], os.path.join(self.out_dir[1], 'non-face'))
		self.save_images(bg_images[train_images:], os.path.join(self.out_dir[0], 'non-face'))

	def generate_face_images(self, persons, num):
		j = 0
		rem = 0 # number of pending images from previous persons which failed to load

		if (len(persons) >= num):
			images_per_person = 1
		else:
			images_per_person = int(len(persons)/num) + 1

		face_images = []

		for idx, p in enumerate(persons):
			processed_images, rem = self.process_person(p, images_per_person+rem)

			face_images += processed_images

			if (len(face_images) >= num):
				rem = 0
				break;

			if (len(face_images) % int(num/10) == 0):
				print "Done loading ", int(len(face_images)*100/num), " % face images"

		if (rem > 0):
			print "Couldn't process all " , num, " images. Failed to load ", rem, " training images"

		return face_images

	def generate_background_images(self, num):
		all_images = list(self.annotations.index)

		background_images = []
		i = 0
		for img in all_images:
			if (i >= num):
				break;

			x, y, w, h = self.get_bounding_box_info(img)

			_bg_img = self.generate_bg_image(img, (x, y, x+w, y+h))
			if (_bg_img is not None):
				background_images.append(_bg_img)
				i += 1

				if (i % int(num/10) == 0):
					print "Done loading ", int(i*100/num), " % bg images"

		return background_images


	def generate_bg_image(self, image, bbox):
		img_path = os.path.join(self.input_dir, image)		
		_img = cv2.imread(img_path)

		height, width, _ = _img.shape
		bg_img = None
		w = self.img_dim[0]
		h = self.img_dim[1]

		if (bbox[0] > w and bbox[1] > h):
			bg_img = _img[0:h, 0:w]
		elif (bbox[0] > w and bbox[3] < height-h):
			bg_img = _img[height-h:height, 0:w]
		elif (bbox[2] < width-w and bbox[3] < height-h):
			bg_img = _img[height-h:height, width-w:width]
		elif (bbox[2] < width-w and bbox[1] > h):
			bg_img = _img[0:h, width-w:width]

		return bg_img


	def process_person(self, p, total):
		i = 0
		t = 0

		randomized_images = os.listdir(os.path.join(self.input_dir, p))
		random.shuffle(randomized_images)

		processed_images = []

		while (i < total and t < len(randomized_images)):
			try:
				image_sample = randomized_images[t]
				img = self.process_image(p, image_sample)
				if img is not None:
					processed_images.append(img)
					i +=1
			except:
				pass
			
			t += 1

		return processed_images, (total - i)

	def process_image(self, person, image):
		if (not self.check_image(person, image)):
			return None
		face_x, face_y, width, height = self.get_bounding_box_info(os.path.join(person, image))

		img_path = os.path.join(self.input_dir, person, image)		
		_img = cv2.imread(img_path)

		if (self.debug_mode and self.debug_count < 10):
			self.draw_bounding_box(_img, face_x, face_y, width, height, image)
			self.debug_count += 1

		resized = self.change_shape_and_size(_img, face_x, face_y, face_x+width, face_y+height)

		return resized

	def check_image(self, person, image):
		t = os.path.join(person, image)
		if (t in self.annotations.index):
			details = self.annotations.loc[t]
			if (int(details['FACE_WIDTH']) < 60 or int(details['FACE_HEIGHT']) < 60):
				return False

			# if (abs(details['ROLL']) > 2 or abs(details['PITCH']) > 2 or abs(details['YAW']) > 2):
			# 	return False

			return True
		else:
			return False

	def change_shape_and_size(self, img, left, top, right, bottom):
		crop_img = img[top:bottom, left:right]
		resized = cv2.resize(crop_img, self.img_dim, interpolation = cv2.INTER_AREA)
		return resized
		

	def get_bounding_box_info(self, image):
		details = self.annotations.loc[image]

		face_x = int(details['FACE_X'])
		face_y = int(details['FACE_Y'])
		width = int(details['FACE_WIDTH'])
		height = int(details['FACE_HEIGHT'])

		return face_x, face_y, width, height


	def draw_bounding_box(self, _img, face_x, face_y, width, height, out_path):
		cv2.rectangle(_img,(face_x,face_y),(face_x+width,face_y+height),(0,255,0))

		out_dir = os.path.join(self.output_dir, 'debug')
		if (not os.path.exists(out_dir)):
			os.makedirs(out_dir)

		cv2.imwrite(os.path.join(out_dir, out_path), _img)

	def save_images(self, images, save_dir):
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)

		i = 0

		for img in images:
			cv2.imwrite(os.path.join(save_dir, str(i)+'.jpg'), img)
			i += 1

if __name__ == '__main__':
	d = DataPreparation('/Users/iankurgarg/Code/Vision/Project-1/umdfaces/umdfaces_batch3', '/Users/iankurgarg/Code/Vision/Project-1/image-classification/images')
	d.run(1000, 100)

