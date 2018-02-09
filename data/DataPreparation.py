import os

import cv2
import math, random

import urllib
import numpy as np


class DataPreparation(object):
	def __init__(self, input_dir, output):
		self.image_dir = input_dir
		self.output_dir = output

		if (not os.path.exists(self.output_dir)):
			os.makedirs(self.output_dir)

		self.train_dir = os.path.join(self.output_dir, 'train')
		self.test_dir = os.path.join(self.output_dir, 'test')

		if (not os.path.exists(self.train_dir)):
			os.makedirs(self.train_dir)

		if (not os.path.exists(self.test_dir)):
			os.makedirs(self.test_dir)


		self.img_dim = (60, 60)

		self.debug_mode = True
		self.debug_count = 0
		self.overall_count = 0
		random.seed(1)

	def generate_train_set(self, persons, num):
		j = 0
		rem = 0 # number of pending images from previous persons which failed to load

		
		if (len(persons) >= num):
			images_per_person = 1
		else:
			images_per_person = int(len(persons)/num) + 1

		print 'images_per_person = ', images_per_person

		for idx, p in enumerate(persons):
			print 'running for person = ', p
			randomized_images = self.get_annotations_for_person(p)
			urls_dict = self.get_image_urls(p)

			print 'parsed files'
			processed_images, rem = self.process_person(p, randomized_images, urls_dict, images_per_person+rem)
			print 'processed images for person'

			for img in processed_images:
				print 'writing file ', j
				cv2.imwrite(os.path.join(self.train_dir, str(j)+'.jpg'), img)
				
				j += 1
				if (j >= num):
					print 'completed all images returning'
					return
			

		if (rem > 0):
			print "Couldn't process all " , num, " images. Failed to load ", rem, " number of training images"

	def process_person(self, p, randomized_images, urls_dict, total):
		i = 0
		t = 0

		processed_images = []

		while (i < total and t < len(randomized_images)):
			print "i = ", i, "t = ", t
			try:
				image_sample = randomized_images[t]
				img = self.process_image(p, image_sample, urls_dict)
				print "proceesed img"
				if (img is not None):
					processed_images.append(img)
					i +=1
			except:
				print "ignore this image  - ", p, " - ", image_sample 

				# pass
			
			t += 1

		return processed_images, (total - i)

	def get_image_urls(self, person):
		folder_path = os.path.join(self.image_dir, person)
		url_file_path = os.path.join(folder_path, 'info.txt')

		with open(url_file_path) as f:
			lines = f.readlines()
			lines = [x for x in lines if len(x.split('\t')) == 3]


		urls_dict = {}
		for x in lines:
			parts = x.split('\t')
			urls_dict[parts[1]] = parts[2]

		return urls_dict


	def get_annotations_for_person(self, person):
		folder_path = os.path.join(self.image_dir, person)
		annotation_file_path = os.path.join(folder_path, 'filelist_LBP.txt')

		with open(annotation_file_path) as f:
			annotations = f.readlines()
			annotations = [x.strip() for x in annotations]

		random.shuffle(annotations)

		return annotations

	def process_image(self, person, image, urls_dict):
		name, left, top, right, bottom = self.get_bounding_box_info(image)
		print "calculated bounding box"
		out_path = os.path.join(person, name)
		# img_path = os.path.join(self.image_dir, out_path)
		img_path = urls_dict[name]
		
		url_response = urllib.urlopen(img_path, timeout=3000)
		if (url_response.geturl() != img_path):
			print "returning because of redirection"
			return None
		
		print "reading from url"
		img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
		_img = cv2.imdecode(img_array, -1)
		
		print "img decoded"
		# _img = cv2.imread(img_path)
		print "here - " + img_path
		if (self.debug_mode and self.debug_count < 10):
			self.draw_bounding_box(_img, left, top, right, bottom, out_path)
			self.debug_count += 1

		# resized = self.change_shape_and_size(_img, left, top, right, bottom)

		# return resized

		return _img

			

	def change_shape_and_size(self, img, left, top, right, bottom):
		crop_img = img[top:bottom, left:right]
		resized = cv2.resize(crop_img, self.img_dim, interpolation = cv2.INTER_AREA)
		return resized

	def run(self, train_images, test_images):
		persons = os.listdir(self.image_dir)
		random.shuffle(persons)
		N = len(persons)
		tN = int(0.8*N)
		train_persons = persons[:tN]
		test_persons = persons[tN:]

		self.generate_train_set(train_persons, train_images)
		# self.generate_test_set(test_persons, test_images)
		

	def get_bounding_box_info(self, line):
		parts = line.split('\t')
		name = parts[0]
		left = int(parts[1])
		top = int(parts[2])
		right = int(parts[3])
		bottom = int(parts[4])

		return name, left, top, right, bottom


	def draw_bounding_box(self, _img, left, top, right, bottom, out_path):
		cv2.rectangle(_img,(left,top),(right,bottom),(0,255,0))

		img_dir, img_name = os.path.split(out_path)
		out_dir = os.path.join(self.output_dir, 'debug', img_dir)
		if (not os.path.exists(out_dir)):
			os.makedirs(out_dir)

		cv2.imwrite(os.path.join(out_dir, img_name), _img)

if __name__ == '__main__':
	d = DataPreparation('/Users/iankurgarg/Code/Vision/Project-1/MSRA/thumbnails_features_deduped_sample', '/Users/iankurgarg/Code/Vision/Project-1/image-classification/images')
	d.run(10, 100)

