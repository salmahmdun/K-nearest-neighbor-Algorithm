from builtins import range     #كرمال نتفادى أي مشاكل بين ال Python versions
from builtins import object    #نفس السطر الأول عشن نعمل object inheritance صح
import numpy as np             #كرمال الحسابات الرياضية

class KNearestNeighbor(object):# class definition
    
    def __init__(self):
        pass
    
    def train(self,X,y):    #training function عشان تحفظ الtraining data
        self.X_train = X
        self.y_train=y
        
    
    def predict(self,X,k=1,loops=0):
        if loops==0:
            distance=self.compute_distance_without_loops(X)
        elif loops==1:
            distance=self.compute_distance_with_one_loops(X)
        elif loops==2:
            distance=self.compute_distance_with_two_loops(X)
        else:
            raise ValueError(" %d is an unavailable as a value for number of loops " %loops)
        
        return self.predict_labels(distance,k=k)
    
    
    
    
    def compute_distance_with_one_loops(self,X):
        num_test_pts = X.shape[0]
        num_train_pts = self.X_train.shape[0]
        distance = np.zeros((num_test_pts, num_train_pts))
        for i in range(num_test_pts):
            distance[i] = np.sqrt(np.sum(np.power(self.X_train - X[i], 2), axis=1))
        return distance
    
    
    
    def compute_distance_with_two_loops(self,X):
        num_test_pts=X.shape[0]
        num_train_pts = self.X_train.shape[0]
        distance = np.zeros((num_test_pts,num_train_pts))
        
        for i in range(num_test_pts):
            for j in range(num_train_pts):
                distance[i,j]=np.sqrt(np.sum(np.power(self.X_train[j]-X[i],2)))
        return distance
    
    
    def compute_distance_without_loops(self,X):
        num_test_pts=X.shape[0]
        num_train_pts=self.X_train.shape[0]
        distance=np.zeros((num_test_pts,num_train_pts))
        
        distance = np.sqrt(-2*(X @ self.X_train.T)+
                           np.power(X,2).sum(axis=1,keepdims=True)+
                           np.power(self.X_train,2).sum(axis=1,keepdims=True).T
                        )
        return distance
    
    


    def predict_labels(self,distance,k=1):
        num_test_pts = distance.shape[0]
        y_predicted = np.zeros(num_test_pts)
        for i in range(num_test_pts):
            nearest_y=[]
            nearest_y = self.y_train[distance[i].argsort()[:k]]
            y_predicted[i] = np.argmax(np.bincount(nearest_y))
        return y_predicted