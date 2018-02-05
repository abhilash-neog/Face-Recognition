from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Load only that data that has min_faces = 40
faces_db = fetch_lfw_people(min_faces_per_person= 40, resize=0.4)

#find out shapes (for plotting)
no_of_samples, h, w = faces_db.images.shape

#X contains data images and features matrix
X = faces_db.data
no_of_features = X.shape[1]

# the label to predict is the id of the person
#class labels
y = faces_db.target

#people names
target_names = faces_db.target_names

no_of_classes = target_names.shape[0]

print(type(no_of_classes),type(target_names[0]))

classes = np.unique(y)#Find the unique elements of an array.
print(classes.shape[0],target_names.shape)
print(classes)


#print dataset details
print("Total dataset size:")
print("no of samples: %d" % no_of_samples)
print("no of features: %d" % no_of_features)
print("no of classes: %d" % no_of_classes)


# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.1, random_state=42)

# Compute a PCA (eigenfaces) on the face dataset
# unsupervised feature extraction / dimensionality reduction
no_of_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (no_of_components, X_train.shape[0]))
#pca = RandomizedPCA(n_components=no_of_components, whiten=True).fit(X_train)
pca = PCA(n_components=no_of_components,svd_solver='randomized',whiten=True).fit(X_train)
#eigen face images - reshape transforms eigen face vectors to eigen face images
eigenfaces = pca.components_.reshape((no_of_components, h, w))

