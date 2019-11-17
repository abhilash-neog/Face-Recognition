# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:20:42 2019
SDS single testing on face recognition 
@author: user
"""
import random
import numpy as np                                         #27,67
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
#from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.metrics import roc_curve, auc
#from scipy import interp
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from time import time
from sklearn.neighbors import KNeighborsClassifier

faces_db = fetch_olivetti_faces()
print("Data Loaded")
no_of_samples, h, w = faces_db.images.shape
print("no of samples: %d" % no_of_samples)
X = faces_db.data
y = faces_db.target

classes = np.unique(y)
classes = ["%d" % number for number in classes]
classes = np.array(classes)
target_names = classes
#y = label_binarize(y, classes=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
n_classes = max(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
no_of_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (no_of_components, X_train.shape[0]))
pca = PCA(n_components=no_of_components, whiten=True).fit(X_train)
eigen_faces = pca.components_
no_eigen_faces,no_features = eigen_faces.shape# isn't both the same????
eigen_faces_lowd = pca.transform(eigen_faces)
#eigen_faces_lowd = eigen_faces.dot(eigen_faces.transpose())
weight_matrix = pca.transform(X_train)
w_matrix = pca.transform(X_test)
#weight_matrix = X_train.dot(eigen_faces.transpose())
print(eigen_faces.shape)
print(eigen_faces_lowd.shape)
print(weight_matrix.shape)
a = weight_matrix.sum(0)#summing each col(sum of the points in dt dimension)
a = a/X_train.shape[0]#avg of each dimension
"""The pcs present have different amt of info/variance of data stored
So basically summing and averaging each dimension means averaging along the components"""
max_eigen_index = np.argmax(a)#max eigen value eigen vector(pc), is the first pc
min_eigen_index = np.argmin(a)#corresponds to lowest pc
print(max_eigen_index, min_eigen_index)
#def obj_function(X,e):
#    X.dot(e.transpose())
N = eigen_faces_lowd.shape[0]
dim = eigen_faces_lowd.shape[1]
V = np.zeros([N,dim])
S = np.zeros([N,dim])#S is the search space(that is why eigen_faces_lowd was created)
np.copyto(S,eigen_faces_lowd)#weight_matrix the set of points
#MaxIt = int(S.shape[0]/3)
MaxIt = 100
print("Maximum Iterations : %d" % MaxIt)
#limit = max(abs(np.amin(S)), abs(np.amax(S)))
up = np.amax(S)
low = np.amin(S)

def CostFun(weight_matrix,e):
    fx = abs(np.sum(e.dot(weight_matrix.transpose())))
    #fx = np.sum(weight_matrix.dot(e.transpose()))
    return fx
def Evaluate(weight_matrix,S):
    fitness = np.zeros([N,1])
    for i in range(0,N):
        fitness[i] = CostFun(weight_matrix,S[i])#S[0] first row,
    return fitness# basically evaluating cost fuc of every data point(fitness stores it)

def Sbound(S,up,low):
    for i in range(0,N):
        if(np.any(S[i]>up) or np.any(S[i]<low)):
            S[i] = np.random.rand()*(up-low)+low
    return S

# =============================================================================
# def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())
#         
# =============================================================================
def thresholdValue(iter,threshold):
    threshold = threshold + np.exp(iter)/(np.exp(iter)+1)#np.log10(i+1)
    return threshold

def calcDisplacement(S,old_S):
    displ = S-old_S
    return displ 

def diffusion(passiv_agents,activ_agents,fitness, S, eigen_faces_lowd):
    found = 0
    for i in range(0,len(activ_agents)):
        ran = random.randint(0,149)#
        for k in range(0,len(activ_agents)):
            if ran==activ_agents[k]:
                found = 1
                break
        if found==1:
            found = 0
            continue
        else:
            fitness[ran] = fitness[i]
            S[ran] = S[i]#check this
            eigen_faces_lowd[ran] = eigen_faces_lowd[i]
             
    return S,eigen_faces_lowd, fitness

def checkAgents(displacement,threshold):
    passiv = []#np.zeros([N,dim])
    activ = []#np.zeros([N,dim])
    
    #hypothesis here is the fitness value
    for i in range(0,N):
        """for k in range(0,len(displacement[0])):
            if np.greater(threshold,displacement[i][k]):
                count+=1"""
        """if count==len(displacement[0]):
            activ[a]=S[i]
            a+=1"""
        if np.greater(threshold,np.average(displacement[i])):
            activ.append(i)
        else:
            passiv.append(i)
            
    return passiv,activ

FBest = LBest = best = bestArg = 0.0
BestChart = np.zeros([MaxIt,1])
fitness = Evaluate(weight_matrix, S)#fitness.shape--150,1
old_S = S

threshold = 0
t1=time()
for i in range(0,MaxIt):
    print("Iteration : %d" % i)
    S = Sbound(S, up, low)
    fitness = Evaluate(weight_matrix, S)
    best = np.max(fitness)
    bestArg = np.argmax(fitness)
    if i==0:
        FBest = best
        LBest = S[bestArg]
    if best>FBest:
        FBest = best
        LBest = S[bestArg]
    
    if i==1:
        threshold = np.average(old_S)
    else:
        threshold = thresholdValue(i,threshold)
        
    displacement = calcDisplacement(S,old_S)
    passive_agents,active_agents = checkAgents(displacement,threshold)
    S, eigen_faces_lowd, fitness = diffusion(passive_agents,active_agents,fitness, S, eigen_faces_lowd)
    
    
t2 = time()
print("time required:",round(t2-t1,3)," s")
print("Iterations completed")

eigen_faces_transformed = pca.inverse_transform(eigen_faces_lowd)
pca.components_ = eigen_faces_transformed
# =============================================================================
# 
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# 
# =============================================================================
#X_train_pca = np.dot(weight_matrix, pca.components_) + pca.mean_
#X_test_pca = np.dot(w_matrix, pca.components_) + pca.mean_

"""parameters = {'C':[1,2,3,4,5,6,7,8,9,10], 'kernel':('linear', 'rbf','poly'), 'gamma':[0.0000001,0.001,0.00005,0.0001,0.00001]}
svc = OneVsRestClassifier(svm.SVC(C = 1.0, kernel = 'poly', gamma = 0.1))
classifier = GridSearchCV(svc, param_grid = parameters)"""
clf = OneVsRestClassifier(svm.SVC(C = 1.0, kernel='rbf',gamma = 0.001, probability=True))

"""param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'kernel':('linear','rbf','poly'),
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }"""
#clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
#clf = OneVsRestClassifier(svm.SVC(C = 20,kernel = 'rbf',gamma = 0.00005,probability=True))
#clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7))
classifier = clf.fit(X_train_pca, y_train)
train_1 = time()
classifier = classifier.fit(X_train_pca, y_train)
train_2 = time()
print("Training time for classifier: ",round(train_2-train_1,3)," s")
pred_1 = time()
y_pred = classifier.predict(X_test_pca)
pred_2 = time()
print("Prediction time for classifier: ",round(pred_2-pred_1,3)," s")
print(classification_report(y_test, y_pred, target_names = target_names))
#y_score = classifier.fit(X_train_pca, y_train).decision_function(X_test_pca)
y_score = classifier.score(X_test_pca, y_test)


