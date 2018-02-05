from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

faces_db = fetch_olivetti_faces()
no_of_samples, h, w = faces_db.images.shape
X = faces_db.data
no_of_features = X.shape[1]
y = faces_db.target
classes = np.unique(y)
classes = ["%d" % number for number in classes]
classes = np.array(classes)
target_names = classes
no_of_classes = classes.shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=42)
no_of_components = 150
pca = PCA(n_components=no_of_components, whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((no_of_components, h, w))