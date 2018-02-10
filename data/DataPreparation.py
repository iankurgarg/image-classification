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

		self.debug_mode = True
		self.debug_count = 0
		self.overall_count = 0
		random.seed(1)

	def generate_images(self, persons, num, train=1):
		j = 0
		rem = 0 # number of pending images from previous persons which failed to load

		if (len(persons) >= num):
			images_per_person = 1
		else:
			images_per_person = int(len(persons)/num) + 1

		for idx, p in enumerate(persons):
			processed_images, rem = self.process_person(p, images_per_person+rem)

			for img in processed_images:
				cv2.imwrite(os.path.join(self.out_dir[train], str(j)+'.jpg'), img)
				
				j += 1
				if (j >= num):
					print 'completed all images returning'
					return
			

		if (rem > 0):
			print "Couldn't process all " , num, " images. Failed to load ", rem, " number of training images"

	def process_person(self, p, total):
		i = 0
		t = 0

		randomized_images = os.listdir(os.path.join(self.input_dir, p))
		random.shuffle(randomized_images)

		processed_images = []

		while (i < total and t < len(randomized_images)):
			# try:
				image_sample = randomized_images[t]
				img = self.process_image(p, image_sample)
				if img is not None:
					processed_images.append(img)
					i +=1
			# except:
			# 	print "ignore this image  - ", p, " - ", image_sample 
			# 	pass
			
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

		# return _img

	def check_image(self, person, image):
		t = os.path.join(person, image)
		if (t in self.annotations.index):
			return True
		else:
			return False

	def change_shape_and_size(self, img, left, top, right, bottom):
		crop_img = img[top:bottom, left:right]
		resized = cv2.resize(crop_img, self.img_dim, interpolation = cv2.INTER_AREA)
		return resized

	def run(self, train_images, test_images):
		persons = os.listdir(self.input_dir)
		persons = [x for x in persons if os.path.isdir(os.path.join(self.input_dir, x))]
		random.shuffle(persons)
		N = len(persons)
		tN = int(0.8*N)
		train_persons = persons[:tN]
		test_persons = persons[tN:]

		self.generate_images(train_persons, train_images, 1)
		self.generate_images(test_persons, test_images, 0)
		

	def get_bounding_box_info(self, image):
		details = self.annotations.loc[image]

		face_x = int(details['FACE_X'])
		face_y = int(details['FACE_Y'])
		width = int(details['FACE_WIDTH'])
		height = int(details['FACE_WIDTH'])

		return face_x, face_y, width, height


	def draw_bounding_box(self, _img, face_x, face_y, width, height, out_path):
		cv2.rectangle(_img,(face_x,face_y),(face_x+width,face_y+height),(0,255,0))

		out_dir = os.path.join(self.output_dir, 'debug')
		if (not os.path.exists(out_dir)):
			os.makedirs(out_dir)

		cv2.imwrite(os.path.join(out_dir, out_path), _img)

if __name__ == '__main__':
	d = DataPreparation('/Users/iankurgarg/Code/Vision/Project-1/umdfaces/umdfaces_batch3', '/Users/iankurgarg/Code/Vision/Project-1/image-classification/images')
	d.run(1000, 100)

