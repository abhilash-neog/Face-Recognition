from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_lfw_people

#db = fetch_lfw_people(min_faces_per_person= 40, resize=0.4)
#print(db.data)
#print(db.target)
lb = LabelBinarizer()
lb.fit([1,2,3,1,2,3,4,5,6,7,8,9,0])#remove ha. prints a col matrix,i.e. when just 2 classes present
#then in transforming a multi label into binary label it shows one class at time
#convert multi-class labels to binary labels (belong or does not belong to the class)
print(lb.classes_)
print(lb.transform([1,2,3,1,2]))