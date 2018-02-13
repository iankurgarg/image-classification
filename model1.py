from sklearn.mixture import GaussianMixture
from data_io.DataLoader import DataLoader




dl = DataLoader("/Users/iankurgarg/Code/Vision/Project-1/image-classification/images")

faces, non_faces = dl.load_data(train=1)

faces = dl.flatten_images(faces)
non_faces = dl.flatten_images(non_faces)


model = GaussianMixture(n_components=1, covariance_type='full')

model.fit(faces)

