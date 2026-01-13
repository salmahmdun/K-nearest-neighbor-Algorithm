from builtins import range     #كرمال نتفادى أي مشاكل بين ال Python versions
from builtins import object    #نفس السطر الأول عشن نعمل object inheritance صح
import numpy as np             #كرمال الحسابات الرياضية



class KNearestNeighbor(object):# class definition
    
    
    def __init__(self):  #constructor
        pass
    
    
    def train(self,X,y):    #training function عشان تحفظ الtraining data   (memorizing training data)
        self.X_train = X    # N.B : No actual training happens here !!!!
        self.y_train=y
        
        
    
    def predict(self,X,k=1,loops=0): 
                                            #X : features 
                                            #k : nb of nearest neighbor 
                                            #loops : nb of loops used to compute distances   
        if loops==0:        # no loops (fastest)
            distance=self.compute_distance_without_loops(X)
        elif loops==1:      # 1 loop (medium speed in calculation)
            distance=self.compute_distance_with_one_loops(X)
        elif loops==2:      # 2 loops (slowest in calculation)
            distance=self.compute_distance_with_two_loops(X)
        else:               # بتعطي error إذا ال loops not in [0,1,2]
            raise ValueError(" %d is an unavailable as a value for number of loops " %loops)
        
        return self.predict_labels(distance,k=k)     #بحدد class(label) لكل test pt
    
    
    
    
    def compute_distance_with_one_loops(self,X): # بنحسب المسافة بين  كل test pt و كل train pt using 1 loop
        num_test_pts = X.shape[0]                # Number of test pts
        num_train_pts = self.X_train.shape[0]   # Number of train pts
        distance = np.zeros((num_test_pts, num_train_pts))  # initialize distance matrix

        for i in range(num_test_pts):
            distance[i] = np.sqrt(np.sum(np.power(self.X_train - X[i], 2), axis=1))         #Ecludian distance formula(L2)

        return distance
    

    
    
    def compute_distance_with_two_loops(self,X):    # بنحسب المسافة بين  كل test pt و كل train pt using 2 loops
        num_test_pts=X.shape[0]               # Number of test pts
        num_train_pts = self.X_train.shape[0]   # Number of train pts
        distance = np.zeros((num_test_pts,num_train_pts))   # initialize distance matrix
        
        for i in range(num_test_pts):               # 1st loop over all test pts
            for j in range(num_train_pts):          # 2nd loop over all train pts
        
                distance[i,j]=np.sqrt(np.sum(np.power(self.X_train[j]-X[i],2)))      #Ecludian distance formula(L2)
        
        return distance
    
    
    
    
    def compute_distance_without_loops(self,X):      # بنحسب المسافة بين  كل test pt و كل train pt without using any loops
        num_test_pts=X.shape[0]         # Number of test pts
        num_train_pts=self.X_train.shape[0]         # Number of train pts
        distance=np.zeros((num_test_pts,num_train_pts))      # initialize distance matrix
        
        distance = np.sqrt(-2*(X @ self.X_train.T)+
                           np.power(X,2).sum(axis=1,keepdims=True)+
                           np.power(self.X_train,2).sum(axis=1,keepdims=True).T
                        )   #Ecludian distance formula(L2) in vectorized form
        
        return distance
    
    


    def predict_labels(self,distance,k=1):      # بتحدد class(label) لكل test pt based on distances matrix
        num_test_pts = distance.shape[0]        # Number of test pts
        y_predicted = np.zeros(num_test_pts)    # initialize predicted labels vector
        
        for i in range(num_test_pts):           # loop over all test pts
            nearest_y=[]             # initialize nearest labels list
            nearest_y = self.y_train[distance[i].argsort()[:k]]     # get the labels of the k nearest neighbors
            y_predicted[i] = np.argmax(np.bincount(nearest_y))      # get the most common label among the k nearest neighbors
        
        return y_predicted