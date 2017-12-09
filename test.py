import pickle
import numpy as np
import os
from sklearn.metrics import classification_report

original_dataset = np.load("./datasets/original_data.npy")
num_face_feature = original_dataset.shape[1] - 1

training_size = 800
X_train = original_dataset[:training_size, :num_face_feature - 1]
X_validation = original_dataset[training_size:, :num_face_feature - 1]
y_train = original_dataset[:training_size, -1]
y_validation = original_dataset[training_size:, -1]


trees = []
model_directory = r"./model"

for filename in os.listdir(model_directory):
    if filename.split(".")[1] == "dot":
        with open(model_directory + "/" + filename, 'rb') as f:
            tree = pickle.load(f)
            trees.append(tree)


alphas = np.load(model_directory + "/alphas.npy")

adaboost_predict = np.zeros(X_validation.shape[0])
count = 0
threshold = 0
weak_classifier_precisions = []
for tree in trees:
    pre = tree.predict(X_validation)
    weak_classifier_precision = np.mean(pre == y_validation)
    weak_classifier_precisions.append(weak_classifier_precision)
    adaboost_predict += alphas[count] * pre
    count += 1

for i in range(len(adaboost_predict)):
    if adaboost_predict[i] > threshold:
        adaboost_predict[i] = 1
    else:
        adaboost_predict[i] = -1


precision = np.mean(adaboost_predict == y_validation)
print(precision)
print(weak_classifier_precisions)
report = classification_report(y_validation, adaboost_predict)
with open("./report.txt", 'w') as f:
    f.write(report)

print(report)