import pickle
import numpy as np
from sklearn import tree


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier_list = []
        self.alpha_list = []
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''

        num_samples, num_features = X.shape
        distribution = np.ones(num_samples) * (1 / num_samples)

        for i in range(self.n_weakers_limit):
            weak_classifier = self.weak_classifier(criterion='entropy', max_depth=8)
            weak_classifier.fit(X, y, sample_weight=distribution)

            self.weak_classifier_list.append(weak_classifier)
            with open("./model/tree_" + str(i+1) + ".dot", 'wb') as f:
                pickle.dump(weak_classifier, f)

            print("Finish No." + str(i+1) + " weak classifier.")
            weak_predict = weak_classifier.predict(X)
            error = np.sum(distribution * (weak_predict != y))
            delta = 1e-6
            alpha = np.log(1. / (error + delta) - 1) / 2.
            self.alpha_list.append(alpha)

            for j in range(num_samples):
                distribution[j] = distribution[j] * np.exp(-alpha * y[j] * weak_predict[j])
            distribution /= (np.sum(distribution) + delta)

        np.save("./model/alphas.npy", np.array(self.alpha_list))
        return




    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''

        predicts = []
        for weak_classifier in self.weak_classifier_list:
            predicts.append(weak_classifier.predict(X))
        predicts = np.array(predicts)

        if predicts.dot(np.array(self.alpha_list)):
            return 1
        else:
            return -1



    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
