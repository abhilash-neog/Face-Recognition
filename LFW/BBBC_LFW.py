import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# =============================================================================
# from sklearn.metrics import roc_curve, auc
# from scipy import interp
from sklearn import svm
# =============================================================================
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import FaceRecognitionLFW

faces_db = fetch_lfw_people(min_faces_per_person= 40, resize=0.4)
print("Data Loaded")
no_of_samples, h, w = faces_db.images.shape
print("no of samples: %d" % no_of_samples)
X = faces_db.data
y = faces_db.target
y = label_binarize(y, classes=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
n_classes = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
no_of_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (no_of_components, X_train.shape[0]))
pca = PCA(n_components=no_of_components, whiten=True).fit(X_train)
eigen_faces = pca.components_
no_eigen_faces,no_features = eigen_faces.shape
eigen_faces_lowd = pca.transform(eigen_faces)
#eigen_faces_lowd = eigen_faces.dot(eigen_faces.transpose())
weight_matrix = pca.transform(X_train)
w_matrix = pca.transform(X_test)
#weight_matrix = X_train.dot(eigen_faces.transpose())
print(eigen_faces.shape)
print(eigen_faces_lowd.shape)
print(weight_matrix.shape)
# =============================================================================
# a = weight_matrix.sum(0)
# a = a/X_train.shape[0]
# max_eigen_index = np.argmax(a)
# min_eigen_index = np.argmin(a)
# print(max_eigen_index, min_eigen_index)
# 
# =============================================================================
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


def CostFun(weight_matrix,e):
    fx = abs(np.sum(e.dot(weight_matrix.transpose())))
    #fx = np.sum(weight_matrix.dot(e.transpose()))
    return fx


def Evaluate(weight_matrix,S):
    fitness = np.zeros([N,1])
    for i in range(0,N):
        fitness[i] = CostFun(weight_matrix,S[i])
    return fitness


def Sbound(S,up,low):
    for i in range(0,N):
        if(np.any(S[i]>up) or np.any(S[i]<low)):
            S[i] = np.random.rand()*(up-low)+low
    return S

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        
def bigCrunch(fitness, S):
    com = np.zeros([1,N])
    for i in range(0,len(S)):
        com[:,i] = np.sum(S[i]/fitness[i])/np.sum(1/fitness[i])
        
    return com

def ChaoticLocalSearch(weight_matrix, centre):
    local_best = CostFun(weight_matrix, centre)
    total_best = local_best
    eigen_local = centre
    eigen_total = eigen_local
    #np.copyto(eigen_local, centre)
    #np.copyto(eigen_total, eigen_local)
    for i in range(0,dim):
        radval = min(abs(centre[:, i]-low), abs(centre[:, i]-up))
        if(i==0):
            minvalue = radval
        if(radval < minvalue):
            minvalue = radval
    r = minvalue
    L = np.zeros([dim,dim])
    chaosM = np.zeros([dim,1])
    #rho = 0.9
    for i in range(0,dim):
        L[i] = centre
        chaosM[i] = NewChaos()
    for num in range(0,25):
        for i in range(0,dim):
            L[i][i] += r*(2*chaosM[i]-1)
            if(CostFun(weight_matrix, L[i]) > local_best):
                local_best = CostFun(weight_matrix, L[i])
                eigen_local = L[i]
            chaosM[i] = 4*chaosM[i]*(1-chaosM[i])
            L[i] = centre
        if local_best > total_best:
            total_best = local_best
            eigen_total = eigen_local
    return eigen_total,total_best

def NewChaos():
    while 1:
        chaos = np.random.rand()
        if chaos != 0.0 and chaos != 0.25 and chaos != 0.5 and chaos != 0.75:
            break
    return chaos

def bigBang(S,com,up,iter,dev):
    S = com + S*np.random.normal(com,dev)/iter
    return S
    
FBest = LBest = best = bestArg = 0.0
BestChart = np.zeros([MaxIt,1])
#fitness = Evaluate(weight_matrix, S)
#M = MassCalc(fitness)
dev = 1

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
    leastArg = np.argmin(fitness)
    least = np.min(fitness)
    print("Least Fitness Value")
    print(least)
    print("Least Fitness Index")
    print(leastArg)
    centre_of_mass = bigCrunch(fitness,S)
    eigen_faces_lowd[leastArg] = centre_of_mass
    eigen_local_best, chaotic_local_best = ChaoticLocalSearch(weight_matrix,centre_of_mass)
    #print("Chaotic local best fitness :")
    #print(chaotic_local_best)
# =============================================================================
    if chaotic_local_best > least:
        fitness[leastArg] = chaotic_local_best
        S[leastArg] = eigen_local_best
        eigen_faces_lowd[leastArg] = eigen_local_best
        print("Eigen vector replaced")
#     M = MassCalc(fitness)
# =============================================================================
    #G = Gconstant(i,MaxIt)
    #a = GField(G, M, S, i, MaxIt, 1)
    
    S = bigBang(S,centre_of_mass,up,i+1,dev-0.04)
    
    
eigen_faces_transformed = pca.inverse_transform(eigen_faces_lowd)
#data_original = np.dot(data_reduced, pca.components_) + pca.mean_
pca.components_ = eigen_faces_transformed

#X_train_pca = pca.transform(X_train)

X_train_pca = np.dot(weight_matrix,pca.components_) + pca.mean_
X_test_pca = np.dot(w_matrix,pca.components_) + pca.mean_
#X_test_pca = pca.transform(X_test)
#C=1.0
#manipulate it to 70%
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
#classifier = OneVsRestClassifier(DecisionTreeClassifier(criterion = 'entropy',min_samples_split= 10))

classifier = classifier.fit(X_train_pca, FaceRecognitionLFW.y_train)

y_pred = classifier.predict(X_test_pca)

print(classification_report(FaceRecognitionLFW.y_test, y_pred, target_names=FaceRecognitionLFW.target_names))

#y_score = classifier.evaluate(X_test_pca)

"""
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
roc = plt.figure(1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linewidth=2)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          linewidth=2)
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linewidth=2)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          linewidth=2)
for i in range(n_classes):
    plt.plot(fpr[i],tpr[i], label=''
                                  ''.format(i, roc_auc[i]))

# for (i, ind) in enumerate(index):
#     plt.plot(fpr[ind], tpr[ind], label='ROC curve of class {0} (area = {1:0.2f})'
#                                    ''.format(ind, roc_auc[ind]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on the ORL Dataset for multiple classes')
plt.legend(loc="lower right")
eigen = plt.figure(2)
eigenface_titles = ["eigenface %d" % i for i in range(eigen_faces.shape[0])]
plot_gallery(eigen_faces, eigenface_titles, h, w)
plt.show()
"""