import os
import cv2
import math

class DataPreparation(object):
	def __init__(self, FDDB, output):
		self.image_dir = os.path.join(FDDB, "originalPics")
		self.folds_dir = os.path.join(FDDB, "FDDB-folds")
		self.output_dir = output

		self.debug_mode = True
		self.debug_count = 0

		if (self.debug_mode):
			self.folds = range(1, 2)
		else:
			self.folds = range(1, 11)


	def process_fold(self, images, annotations):
		j = 0

		for im in images:
			im += '.jpg'
			assert im == annotations[j], "order of images and annotations is not same"

			j += 1
			number_of_faces = int(annotations[j])
			j += 1

			for i in range(number_of_faces):
				r1, r2, angle, x, y = self.get_ellipse_info(annotations[j])
				j += 1

				if (self.debug_mode == True and self.debug_count < 10):
					self.draw_bounding_box(im, r1, r2, angle, x, y)

				print "Image = ", im, r1, r2, angle, x, y





	def run(self, train_images, test_images):

		for i in folds:
			image_set_file = "FDDB-fold-" + "%02d" % (i,) + ".txt"
			annotation_file = "FDDB-fold-" + "%02d" % (i,) + "-ellipseList.txt"

			with open(os.path.join(self.folds_dir, image_set_file)) as f:
				images = f.readlines()
				images = [x.strip() for x in images]
			
			with open(os.path.join(self.folds_dir, annotation_file)) as f:
				annotations = f.readlines()
				annotations = [x.strip() for x in annotations]


			self.process_fold(images, annotations)

	def get_ellipse_info(self, line):
		parts = line.split(' ')
		major_axis_radius = float(parts[0])
		minor_axis_radius = float(parts[1])
		angle = float(parts[2])
		center_x = float(parts[3])
		center_y = float(parts[4])

		return major_axis_radius, minor_axis_radius, angle, center_x, center_y


	def draw_bounding_box(self, img, r1, r2, angle, x, y):
		img_path = os.path.join(self.image_dir, img)
		_img = cv2.imread(img_path)

		cv2.ellipse(_img, (x, y), (r1, r2), angle, 0, 360, 255, -1)

		img_dir, img_name = os.path.split(img)
		out_dir = os.path.join(self.output_dir, 'debug', img_dir)
		if (not os.path.exists(out_dir)):
			os.makedirs(out_dir)

		cv2.imwrite(os.path.join(out_dir, img_name))

