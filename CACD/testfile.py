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
eigen_faces = pca.components_
no_eigen_faces,no_features = eigen_faces.shape
eigen_faces_lowd = pca.transform(eigen_faces)
#eigen_faces_lowd = eigen_faces.dot(eigen_faces.transpose())
weight_matrix = pca.transform(X_train)
#weight_matrix = X_train.dot(eigen_faces.transpose())
print(eigen_faces.shape)
print(eigen_faces_lowd.shape)
print(weight_matrix.shape)
a = weight_matrix.sum(0)
a = a/X_train.shape[0]
max_eigen_index = np.argmax(a)
min_eigen_index = np.argmin(a)
print(max_eigen_index, min_eigen_index)
#def obj_function(X,e):
#    X.dot(e.transpose())
N = eigen_faces_lowd.shape[0]
dim = eigen_faces_lowd.shape[1]
V = np.zeros([N,dim])
S = np.zeros([N,dim])
np.copyto(S,eigen_faces_lowd)
MaxIt = int(S.shape[0]/3)
print("Maximum Iterations : %d" % MaxIt)
#limit = max(abs(np.amin(S)), abs(np.amax(S)))
up = np.amax(S)
low = np.amin(S)