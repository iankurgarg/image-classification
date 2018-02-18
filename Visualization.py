from matplotlib import pyplot as plt
import numpy as np

def show_mean(faces):
	mean_face = np.mean(faces, axis=0)
	plt.imshow(mean_face.reshape((60,60)), cmap="gray");
	plt.show();

def show_cov(faces):
	cov_face = np.cov(faces, rowvar=False)
	cov_diag = np.diag(cov_face)
	plt.imshow(cov_diag.reshape((60,60)), cmap="gray");
	plt.show();