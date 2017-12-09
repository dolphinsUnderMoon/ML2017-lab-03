import numpy as np
from sklearn.cross_validation import train_test_split
from ensemble import *
from sklearn import tree

if __name__ == "__main__":
    original_dataset = np.load("./datasets/original_data.npy")
    num_face_feature = original_dataset.shape[1] - 1

    training_size = 800
    X_train = original_dataset[:training_size, :num_face_feature - 1]
    X_validation = original_dataset[training_size:, :num_face_feature - 1]
    y_train = original_dataset[:training_size, -1]
    y_validation = original_dataset[training_size:, -1]

    print("DATASET IS READY")
    adaboost_classifier = AdaBoostClassifier(tree.DecisionTreeClassifier, 10)
    adaboost_classifier.fit(X_train, y_train)