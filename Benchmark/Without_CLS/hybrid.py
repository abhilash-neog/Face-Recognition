import numpy as np                                         #27,67
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
#import FaceRecognition
import random
from time import time

N = 0
dim = 0
def face_extract():
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
	no_eigen_faces,no_features = eigen_faces.shape# isn't both the same????
	eigen_faces_lowd = pca.transform(eigen_faces)
	#eigen_faces_lowd = eigen_faces.dot(eigen_faces.transpose())
	weight_matrix = pca.transform(X_train)
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
	#   	 X.dot(e.transpose())	
	N = eigen_faces_lowd.shape[0]
	dim = eigen_faces_lowd.shape[1]
	V = np.zeros([N,dim])
	S = np.zeros([N,dim])#S is the search space(that is why eigen_faces_lowd was created)
	np.copyto(S,eigen_faces_lowd)#weight_matrix the set of points
	#MaxIt = int(S.shape[0]/3)
	MaxIt = 50
	print("Maximum Iterations : %d" % MaxIt)
	#limit = max(abs(np.amin(S)), abs(np.amax(S)))
	up = np.amax(S)
	low = np.amin(S)
	return S, weight_matrix

def CostFun(weight_matrix,e):
    fx = abs(np.sum(e.dot(weight_matrix.transpose())))
    #fx = np.sum(weight_matrix.dot(e.transpose()))
    return fx


def Evaluate(weight_matrix,S):
    fitness = np.zeros([150,1])
    for i in range(0,150):
        fitness[i] = CostFun(weight_matrix,S[i])
    return fitness
#length of fitness is 150,whereas total points 360??

def MassCalc(fitness):
    Fmax = max(fitness)
    Fmin = min(fitness)
    if Fmax == Fmin:
        M = np.ones([150,1])
    else:
        worst = Fmin
        best = Fmax
        m = (fitness-worst)/(best-worst)
        M = m/np.sum(m)
    return M


def Gconstant(iteration,MaxIt):
    alpha = 20.0
    G0 = 100.0
    G=G0*np.exp(-alpha*(float(iteration)/float(MaxIt)))
    return G


def GField(G,M,S,iteration,MaxIt,Elitcheck,N,dims):
    Finalper = 2.0
    if Elitcheck==1:
        kbest = Finalper+(1.0-float(iteration)/float(MaxIt))*(100-Finalper)
        kbest = np.round(N*kbest/100)
    else:
        kbest = N
    F = np.zeros([N,dims])
    MAsort = np.argsort(M, axis=0)
    for i in range(0,N):
        for ii in range(1,np.int(kbest)):
            j = MAsort[-(ii)]
            if j!=i:
                R = np.linalg.norm((S[i]-S[j]))
                chaos = NewChaos()
                for k in range(0,N):
                    F[i,k] += chaos*M[j]*((S[j, k]-S[i, k])/R)
                    chaos = 4*chaos*(1-chaos)
                    if chaos == 0.0 or chaos == 0.25 or chaos == 0.5 or chaos == 0.75 or chaos == 1.0:
                        chaos = NewChaos()
    a = F*G
    #print(a)
    return a


def move(S,V):
    chaos = NewChaos()
    v = np.zeros([V.shape[0], V.shape[1]])
    #print(V.shape)

    for i in range(0,V.shape[0]):
        v[i] = chaos*V[i]#a[i] added the accleration matrix
        chaos = 4*chaos*(1-chaos)
        if chaos == 0.0 or chaos == 0.25 or chaos == 0.5 or chaos == 0.75 or chaos == 1.0:
            chaos = NewChaos()
    s = S + v
    return s,v


def Sbound(S,up,low):
    for i in range(0,N):
        if(np.any(S[i]>up) or np.any(S[i]<low)):
            S[i] = np.random.rand()*(up-low)+low
    return S


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
        
def checkAgents(displacement,threshold, N):
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

def diffusion(passiv_agents,activ_agents,fitness):
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
             
    return fitness

def thresholdValue(iter,threshold):
    threshold = threshold + np.log10(iter)
    return threshold

def calcDisplacement(S,old_S):
    displ = S-old_S
    return displ    

"""def main():
	S, weight_matrix, V, MaxIt, up, low = face_extract()
	FBest = LBest = best = bestArg = 0.0
	BestChart = np.zeros([MaxIt,1])
	fitness = Evaluate(weight_matrix, S)
	old_S = S
	#Displacement for the first iteration
	M = MassCalc(fitness)
	G = Gconstant(0,MaxIt)
	a = GField(G, M, S, 0, MaxIt, 1)
	S, V = move(S, a, V)
	M = MassCalc(fitness)

	for i in range(1,MaxIt):
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
    	fitness = diffusion(passive_agents,active_agents,fitness)
        
    	leastArg = np.argmin(fitness)
        least = np.min(fitness)#finding the least on the updated list
    	print("Least Fitness Value")
     	print(least)
      	print("Least Fitness Index")
    	print(leastArg)
    	mass_sum = np.sum(M)
     	centre_of_mass = (M.transpose().dot(S))/mass_sum
     	eigen_local_best, chaotic_local_best = ChaoticLocalSearch(weight_matrix,centre_of_mass)
    	print("Chaotic local best fitness :")
     	print(chaotic_local_best)
     	if chaotic_local_best > least:
        	S[leastArg] = eigen_local_best
        	eigen_faces_lowd[leastArg] = eigen_local_best
        	print("Eigen vector replaced")
    	old_S = S
     	M = MassCalc(fitness)
    	G = Gconstant(i,MaxIt)
    	a = GField(G, M, S, i, MaxIt, 1)
     	S, V = move(S, a, V)
	t2 = time()
	print("time required:",round(t2-t1,3))
	eigen_faces_transformed = pca.inverse_transform(eigen_faces_lowd)
	pca.components_ = eigen_faces_transformed
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	classifier = OneVsRestClassifier(svm.SVC(kernel='linear', gamma = 0.000001, probability=True))
	classifier = classifier.fit(X_train_pca, FaceRecognition.y_train)
	y_pred = classifier.predict(X_test_pca)
	print(classification_report(FaceRecognition.y_test, y_pred, target_names=FaceRecognition.target_names))
	y_score = classifier.fit(X_train_pca, y_train).decision_function(X_test_pca)
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
if __name__== "__main__":
    main()
else:
    print("Go")


