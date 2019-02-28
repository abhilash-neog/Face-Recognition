# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:45:00 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import FaceRecognition

faces_db = fetch_olivetti_faces()
print("Data Loaded")
no_of_samples, h, w = faces_db.images.shape
print("no of samples: %d" % no_of_samples)
X = faces_db.data
y = faces_db.target
y = label_binarize(y, classes=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
n_classes = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
no_of_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (no_of_components, X_train.shape[0]))
pca = PCA(n_components=no_of_components, whiten=True).fit(X_train)
#eigen_faces_lowd = eigen_faces.dot(eigen_faces.transpose())
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
classifier = classifier.fit(X_train_pca, FaceRecognition.y_train)
y_pred = classifier.predict(X_test_pca)
print(classification_report(FaceRecognition.y_test, y_pred, target_names=FaceRecognition.target_names))
y_score = classifier.fit(X_train_pca, y_train).decision_function(X_test_pca)
