import numpy as np

face_features = np.load("./datasets/face_features.npy")
nonface_features = np.load("./datasets/nonface_features.npy")

num_face_sample, num_face_feature = face_features.shape
num_nonface_sample, num_nonface_feature = nonface_features.shape

positive_label = [np.ones(1) for i in range(num_face_sample)]
negative_label = [-np.ones(1) for i in range(num_nonface_sample)]

positive_samples = np.concatenate((face_features, positive_label), axis=1)
negative_samples = np.concatenate((nonface_features, negative_label), axis=1)

original_dataset = np.concatenate((positive_samples, negative_samples), axis=0)

np.random.shuffle(original_dataset)
np.save("./datasets/original_data.npy", original_dataset)