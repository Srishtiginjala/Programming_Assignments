#IC272LAB6
#Name         :Srishti Ginjala
#Rollno       :B19084
#Mobile Number:9440000900

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Question1
df=pd.read_csv(r"C:\Users\srish\Downloads\mnist-tsne-train.csv")
train_data=df.drop(['labels'],axis=1)
train=train_data.values
true1=df['labels'].values
from sklearn.cluster import KMeans
K = 10
kmeans = KMeans(n_clusters=K)
kmeans.fit(train_data)
kmeans_prediction = kmeans.predict(train_data)
centres=kmeans.cluster_centers_
plt.scatter(train[:, 0], train[:, 1], c=kmeans_prediction,cmap='viridis')
plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
plt.title("Clusters(kmeans) of train data")
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.show()
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
def purity_score(y_true, y_pred):
     contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
     # Find optimal one-to-one mapping between cluster labels and true labels
     row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
     # Return cluster accuracy
     return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
score=purity_score(true1,kmeans_prediction)
print("Purity score(kmeans) for train data is ",score)
df1=pd.read_csv(r"C:\Users\srish\Downloads\mnist-tsne-test.csv")
test_data=df1.drop(['labels'],axis=1)
test=test_data.values
true2=df1['labels'].values
y_pred=kmeans.predict(test_data)
plt.scatter(test[:, 0], test[:, 1], c=y_pred,cmap='viridis')
plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
plt.title("Clusters(kmeans) of test data")
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.show()
print("\nThe confusion matrix(kmeans) for test data is:\n")
score1=purity_score(true2,y_pred)
print("Purity score(kmeans) for test data is ",score1)

#Question2
from sklearn.mixture import GaussianMixture
K = 10
gmm = GaussianMixture(n_components = K)
gmm.fit(train_data)
GMM_prediction = gmm.predict(train_data)
centres=gmm.means_
plt.scatter(train[:, 0], train[:, 1], c=GMM_prediction,cmap='viridis')
plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
plt.title("Clusters(GMM) of train data")
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.show()
print("\nThe confusion matrix(GMM) for train data is:\n")
score=purity_score(true1,GMM_prediction)
print("Purity score(GMM) for train data is ",score)
y_pred=gmm.predict(test_data)
plt.scatter(test[:, 0], test[:, 1], c=y_pred,cmap='viridis')
plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
plt.title("Clusters(GMM) of test data")
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.show()
print("\nThe confusion matrix(GMM) for test data is:\n")
score1=purity_score(true2,y_pred)
print("Purity score(GMM) for test data is ",score1)

#Question3
from sklearn.cluster import DBSCAN
dbscan_model=DBSCAN(eps=5, min_samples=10).fit(train_data)
DBSCAN_predictions = dbscan_model.labels_
plt.scatter(train[:, 0], train[:, 1], c=DBSCAN_predictions,cmap='viridis')
plt.title("Clusters(DBSCAN) of train data")
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.show()
print("\nThe confusion matrix(DBSCAN) for train data is:\n")
score=purity_score(true1,DBSCAN_predictions)
print("Purity score(DBSCAN) for train data is ",score)
from scipy import spatial
def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j]=dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new
dbtest = dbscan_predict(dbscan_model,test, metric =spatial.distance.euclidean)
plt.scatter(test[:, 0], test[:, 1], c=dbtest,cmap='viridis')
plt.title("Clusters(DBSCAN) of test data")
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.show()
print("\nThe confusion matrix(DBSCAN) for test data is:\n")
score1=purity_score(true2,dbtest)
print("Purity score(DBSCAN) for test data is ",score1)

#bonus questions
def kmean(K):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(train_data)
    kmeans_prediction = kmeans.predict(train_data)
    centres=kmeans.cluster_centers_
    plt.scatter(train[:, 0], train[:, 1], c=kmeans_prediction,cmap='viridis')
    plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
    plt.title("Clusters(kmeans) of train data")
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.show()
    def purity_score(y_true, y_pred):
         contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
         # Find optimal one-to-one mapping between cluster labels and true labels
         row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
         # Return cluster accuracy
         return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
    score=purity_score(true1,kmeans_prediction)
    print("Purity score(kmeans) for train data is ",score)
    df1=pd.read_csv(r"C:\Users\srish\Downloads\mnist-tsne-test.csv")
    test_data=df1.drop(['labels'],axis=1)
    test=test_data.values
    true2=df1['labels'].values
    y_pred=kmeans.predict(test_data)
    plt.scatter(test[:, 0], test[:, 1], c=y_pred,cmap='viridis')
    plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
    plt.title("Clusters(kmeans) of test data")
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.show()
    print("\nThe confusion matrix(kmeans) for test data is:\n")
    score1=purity_score(true2,y_pred)
    print("Purity score(kmeans) for test data is ",score1)
kmean(2)
kmean(5)
kmean(8)
kmean(12)
kmean(18)
kmean(20)

distortions = [] 
mapping1 = {} 
K = [2,5,8,10,12,18,20]
from scipy.spatial.distance import cdist 
X=train_data
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(X) 
    kmeanModel.fit(X)       
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'),axis=1)) / X.shape[0])   
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'),axis=1)) / X.shape[0] 
plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 

def gaus(k):
    gmm = GaussianMixture(n_components = k)
    gmm.fit(train_data)
    GMM_prediction = gmm.predict(train_data)
    centres=gmm.means_
    plt.scatter(train[:, 0], train[:, 1], c=GMM_prediction,cmap='viridis')
    plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
    plt.title("Clusters(GMM) of train data")
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.show()
    def purity_score(y_true, y_pred):
         contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
         # Find optimal one-to-one mapping between cluster labels and true labels
         row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
         # Return cluster accuracy
         return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
    score=purity_score(true1,GMM_prediction)
    print("Purity score(GMM) for train data is ",score)
    y_pred=gmm.predict(test_data)
    plt.scatter(test[:, 0], test[:, 1], c=y_pred,cmap='viridis')
    plt.scatter(centres[:,0],centres[:,1],color='red',marker="^")
    plt.title("Clusters(GMM) of test data")
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.show()
    print("\nThe confusion matrix(GMM) for test data is:\n")
    score1=purity_score(true2,y_pred)
    print("Purity score(GMM) for test data is ",score1)
gaus(2)  
gaus(5)  
gaus(8)  
gaus(12)  
gaus(18)  
gaus(20)  

distortions = [] 
mapping1 = {} 
K = [2,5,8,10,12,18,20]
from scipy.spatial.distance import cdist 
X=train_data
for k in K: 
    #Building and fitting the model 
    gmm = GaussianMixture(n_components = k)
    gmm.fit(train_data)
    centres=gmm.means_      
    distortions.append(sum(np.min(cdist(X, centres,'euclidean'),axis=1)) / X.shape[0])   
    mapping1[k] = sum(np.min(cdist(X,centres,'euclidean'),axis=1)) / X.shape[0] 
plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 

def db(ep,sam):
    dbscan_model=DBSCAN(eps=ep, min_samples=sam).fit(train_data)
    DBSCAN_predictions = dbscan_model.labels_
    plt.scatter(train[:, 0], train[:, 1], c=DBSCAN_predictions,cmap='viridis')
    plt.title("Clusters(DBSCAN) of train data")
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.show()
    score=purity_score(true1,DBSCAN_predictions)
    print("Purity score(DBSCAN) for train data is ",score)
    from scipy import spatial
    def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
        # Result is noise by default
        y_new = np.ones(shape=len(X_new), dtype=int)*-1 
        # Iterate all input samples for a label
        for j, x_new in enumerate(X_new):
            # Find a core sample closer than EPS
            for i, x_core in enumerate(dbscan_model.components_):
                if metric(x_new, x_core) < dbscan_model.eps:
                    # Assign label of x_core to x_new
                    y_new[j]=dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                    break
        return y_new
    dbtest = dbscan_predict(dbscan_model,test, metric =spatial.distance.euclidean)
    plt.scatter(test[:, 0], test[:, 1], c=dbtest,cmap='viridis')
    plt.title("Clusters(DBSCAN) of test data")
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.show()
    print("\nThe confusion matrix(DBSCAN) for test data is:\n")
    score1=purity_score(true2,dbtest)
    print("Purity score(DBSCAN) for test data is ",score1)
db(1,10)
db(10,10)
db(5,1)
db(5,30)
db(5,50)

 
