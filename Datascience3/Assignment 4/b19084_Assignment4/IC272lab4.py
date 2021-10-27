#IC272LAB3
#Name         :Srishti Ginjala
#Rollno       :B19084
#Mobile Number:9440000900

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

df=pd.read_csv("seismic_bumps1.csv")
df.drop(['nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89'],axis=1,inplace=True)
grouped=df.groupby('class')
df_1=grouped.get_group(1)
df_0=grouped.get_group(0)
y1=df_1['class']
y0=df_0['class']
df_1.drop(['class'],axis=1,inplace=True)
df_0.drop(['class'],axis=1,inplace=True)

#Question1
[df_train1,df_test1,df_label_train1,df_label_test1] =train_test_split(df_1,y1,test_size=0.3,random_state=42,shuffle=True)
[df_train0,df_test0,df_label_train0,df_label_test0] =train_test_split(df_0,y0,test_size=0.3,random_state=42,shuffle=True)
df_train=pd.concat([df_train0,df_train1])
df_label_train=pd.concat([df_label_train0,df_label_train1])
df_test=pd.concat([df_test0,df_test1])
df_label_test=pd.concat([df_label_test0,df_label_test1])
from sklearn.neighbors import KNeighborsClassifier
#k=1
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(df_train,df_label_train)
pred=neigh.predict(df_test)
from sklearn.metrics import confusion_matrix
arr1= confusion_matrix(df_label_test,pred)
print("\nFor k=1:")
print("The confusion matrix is:")
print(arr1)
from sklearn.metrics import accuracy_score
print("The accuracy_score is:")
print(round(accuracy_score(df_label_test,pred),3))

#k=3
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(df_train,df_label_train)
pred=neigh.predict(df_test)
from sklearn.metrics import confusion_matrix
arr3 = confusion_matrix(df_label_test,pred)
print("\nFor k=3:")
print("The confusion matrix is:")
print(arr3)
from sklearn.metrics import accuracy_score
print("The accuracy_score is:")
print(round(accuracy_score(df_label_test,pred),3))

#k=5
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(df_train,df_label_train)
pred=neigh.predict(df_test)
from sklearn.metrics import confusion_matrix
arr5 = confusion_matrix(df_label_test,pred)
print("\nFor k=5:")
print("The confusion matrix is:")
print(arr5)
from sklearn.metrics import accuracy_score 
print("The accuracy_score is:")
print(round(accuracy_score(df_label_test,pred),3))

print("\nThe value of accuracy is highest for k=5")

#Question2
att=['seismic','seismoacoustic','shift','genergy','gpuls','gdenergy','gdpuls','ghazard','energy','maxenergy']
df_train_norm=df_train.copy()
df_test_norm=df_test.copy()

for i in att:
    df_test_norm[i]=(df_test_norm[i]-df_train_norm[i].min())/(df_train_norm[i].max()-df_train_norm[i].min())
    df_train_norm[i]=(df_train_norm[i]-df_train_norm[i].min())/(df_train_norm[i].max()-df_train_norm[i].min())

#k=1
neigh1 = KNeighborsClassifier(n_neighbors=1)
neigh1.fit(df_train_norm,df_label_train)
pred1=neigh1.predict(df_test_norm)
from sklearn.metrics import confusion_matrix
arr1=confusion_matrix(df_label_test,pred1)
print("\nFor k=1:")
print("The confusion matrix is:")
print(arr1)
from sklearn.metrics import accuracy_score
print("The accuracy_score is:")
print(round(accuracy_score(df_label_test,pred1),3))

#k=3
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(df_train_norm,df_label_train)
pred2=neigh.predict(df_test_norm)
from sklearn.metrics import confusion_matrix
arr2 = confusion_matrix(df_label_test,pred2)
print("\nFor k=3:")
print("The confusion matrix is:")
print(arr2)
from sklearn.metrics import accuracy_score
print("The accuracy_score is:")
print(round(accuracy_score(df_label_test,pred2),3))

#k=5
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(df_train_norm,df_label_train)
pred3=neigh.predict(df_test_norm)
from sklearn.metrics import confusion_matrix
arr3 = confusion_matrix(df_label_test,pred3)
print("\nFor k=5:")
print("The confusion matrix is:")
print(arr3)
from sklearn.metrics import accuracy_score
print("The accuracy_score is:")
print(round(accuracy_score(df_label_test,pred3),3))
print("\nThe value of accuracy is highest for k=5")

#Question3
test=df_test.to_numpy()
m1=np.array(df_train1.mean())
m0=np.array(df_train0.mean())
c1=df_train1.cov().to_numpy()
c0=df_train0.cov().to_numpy()
prior1=len(df_train1)/(len(df_train0)+len(df_train1))
prior0=len(df_train0)/(len(df_train0)+len(df_train1))

def prob(x,m,c):
    x_mu=x-m
    mahalanobis=np.linalg.multi_dot([x_mu.T,np.linalg.inv(c),x_mu])
    expo=math.exp(-mahalanobis/2)
    return(expo/((np.abs(np.linalg.det(c)))**0.5)*((2*math.pi)**5))   
test_pred=[]
for i in range(len(test)):
    x=test[i]
    plike1=prob(x,m1,c1)
    plike0=prob(x,m0,c0)
    pc0=(plike0*prior0)/(plike0*prior0+plike1*prior1)
    pc1=(plike1*prior1)/(plike0*prior0+plike1*prior1)
    if(pc0>pc1):
        test_pred.append(0)
    else:
        test_pred.append(1)
from sklearn.metrics import confusion_matrix
arr3 = confusion_matrix(df_label_test,test_pred)
print("\nThe confusion matrix for Bayes Classifier is:")
print(arr3)
from sklearn.metrics import accuracy_score
print("The accuracy_score is:")
print(round(accuracy_score(df_label_test,test_pred),3))
