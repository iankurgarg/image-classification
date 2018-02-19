from matplotlib import pyplot as plt
import numpy as np

def show_mean(faces, dim3=1):
	mean_face = np.mean(faces, axis=0).astype(np.uint8)
	if dim3 == 1:
		dims = (60,60)
	else:
		dims = (60,60, dim3)
	plt.imshow(mean_face.reshape(dims), cmap="gray");
	plt.show();

def show_cov(faces, dim3=1):
	faces2 = faces.astype(np.float32)/256

	cov_face = np.cov(faces, rowvar=False)
	cov_diag = np.diag(cov_face)
	if dim3 == 1:
		dims = (60,60)
	else:
		dims = (60,60, dim3)
	plt.imshow(cov_diag.reshape(dims), cmap="gray");
	plt.show();

def ROV(fpr, tpr):
	plt.plot(fpr, tpr)