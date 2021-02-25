### Module imports ###
import os
import json
from datetime import datetime
import itertools
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gensim import corpora, matutils
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from load_from_file import FileReader, FileStore
from preprocess_text import FeatureExtraction
import config as cf


### Global Variables ###


### Class declarations ###
class Classifier(object):

    def __init__(self, features_train = None, labels_train = None, features_test = None, labels_test = None, estimator=None):
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.estimator = estimator

    def training(self):
        print('Tranning... ', str(datetime.now()))
        self.estimator.fit(self.features_train, self.labels_train)
        self.training_result()
        print('Tranning Done! ', str(datetime.now()))

    def save_model(self, file_path):
        print('Saving Model... ', str(datetime.now()))
        FileStore(file_path=file_path).save_pickle(obj=self.estimator)
        print('Save Model Done! ', str(datetime.now()))

    def training_result(self):
        y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
        cnf_matrix = confusion_matrix(y_true, y_pred)
        # plot_confusion_matrix(cnf_matrix, title='Confusion matrix from Logistic')
        print('Accuracy: ', self.estimator.score(self.features_test, self.labels_test))
        print(type(self.labels_test), type(self.features_test))
        print(classification_report(y_true, y_pred))


### Function declarations ###
def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', folder_path='data/train'):
    classes = os.listdir(folder_path)
    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='tfidf', help='type of feature')
    parser.add_argument('--model', type=str, default='logistic', help='type of model')
    args = parser.parse_args()

    # Read feature extraction
    print('Reading Feature Extraction... ', str(datetime.now()))
    if args.feature == 'bow':
        features_train_path = f'{cf.FEATURES_TRAIN}_bow.pkl'
        features_test_path = f'{cf.FEATURES_TEST}_bow.pkl'
    elif args.feature == 'tfidf':
        features_train_path = f'{cf.FEATURES_TRAIN}_tfidf.pkl'
        features_test_path = f'{cf.FEATURES_TEST}_tfidf.pkl'

    features_test_loader = pickle.load(open(features_train_path, 'rb'))
    features_train_loader = pickle.load(open(features_test_path, 'rb'))
    features_train, labels_train = FeatureExtraction(data=features_train_loader).read_feature()
    features_test, labels_test = FeatureExtraction(data=features_test_loader).read_feature()
    print('Read Feature Extraction Done! ', str(datetime.now()))

    # Load and train model
    if args.model == 'knn':
        # KNeighbors Classifier
        print('Training by KNeighbors Classifier ...')
        estKNeighbors = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test, estimator=KNeighborsClassifier(n_neighbors=3, n_jobs=4))
        estKNeighbors.training()
        estKNeighbors.save_model(file_path=f'pretrained_model/knn_model_{args.feature}.pkl') # save Model
        print('Training by KNeighbors Classifier Done!')

    elif args.model == 'naive_bayes':
        # Naive Bayes Classifier
        print('Training by Naive Bayes Classifier ...')
        estSVM = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test, estimator= MultinomialNB())
        estSVM.training()
        estSVM.save_model(file_path=f'pretrained_model/svm_model_{args.feature}.pkl') # save Model
        print('Training by SVM Classifier Done!')


    elif args.model == 'svm':
        # SVM Classifier 
        print('Training by SVM Classifier ...')
        estSVM = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test, estimator= LinearSVC(penalty='l2', C=4))
        estSVM.training()
        estSVM.save_model(file_path=f'pretrained_model/svm_model_{args.feature}.pkl') # save Model
        print('Training by SVM Classifier Done!')

    elif args.model == 'randomforest':
        # RandomForest Classifier
        print('Training by RandomForest Classifier ...')
        estRandomForest = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test, estimator=RandomForestClassifier(n_jobs=4))
        estRandomForest.training()
        estRandomForest.save_model(file_path=f'pretrained_model/random_forest_model_{args.feature}.pkl') # save Model
        print('Training by RandomForest Classifier Done!')

    else:
        raise ValueError("Model type is incorrect")
