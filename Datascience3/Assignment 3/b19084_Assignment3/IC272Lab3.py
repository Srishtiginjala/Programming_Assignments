#IC272LAB3
#Name         :Srishti Ginjala
#Rollno       :B19084
#Mobile Number:9440000900


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("landslide_data3.csv")

#Question1
att=["temperature","humidity","rain","pressure","moisture","lightmax","lightavgw/o0"]
df_n=pd.read_csv("landslide_data3.csv")
for i in att:
    q1=np.percentile(df_n[i],25)
    q3=np.percentile(df_n[i],75)
    mid=df_n[i].median()
    #replacing outliers with median
    df_n[i]=np.where((df_n[i]>(2.5*q3-1.5*q1)) | (df_n[i]<(2.5*q1-1.5*q3)),mid,df_n[i])
    #rescaling or normalizing it to (3,6)
    l=df_n[i].min()
    u=df_n[i].max()
    print("\nBefore normalizing %s"%i)
    print("Minimum:%0.3f"%l)
    print("Maximum:%0.3f"%u)
    df_n[i]=3+(df_n[i]-l)*6/(u-l)
    print("\nAfter normalizing %s"%i)
    print("Minimum:%0.3f"%df_n[i].min())
    print("Maximum:%0.3f"%df_n[i].max())
    #standardizing the attributes
    me=df[i].mean()
    st=df[i].std()
    print("\nBefore standardizing %s"%i)
    print("Mean:%0.3f"%me)
    print("Standard Deviation:%0.3f"%st)
    df[i]=(df[i]-me)/st
    print("\nAfter standardizing %s"%i)
    print("Mean:%0.3f"%df[i].mean())
    print("Standard Deviation:%0.3f"%df[i].std())

#Question2
    #2a)
mean=[0,0]
cov=[[6.84806467,7.63444163],[7.63444163,13.02074623]]
d=np.random.multivariate_normal(mean,cov,1000)
plt.scatter(d[:,0],d[:,1],marker='x')
plt.axis('equal')
plt.xticks(range(-15, 20, 5))
plt.yticks(range(-15, 20, 5))
plt.title('scatter plot',fontsize=15)
plt.xlabel('x1',fontsize=18)
plt.ylabel('y1',fontsize=18)
plt.show(False)

val,vec=np.linalg.eig(cov)
print('eigen values are:%0.3f,%0.3f'%(val[0],val[1]))
print('Eigen vectors are:')
print((vec))


plt.quiver(0,0,vec[0,1],vec[1,1],scale=3)
plt.quiver(0,0,vec[0,0],vec[1,0],scale=6)
plt.xticks(range(-20, 20, 5),fontsize=12)
plt.yticks(range(-15, 20, 5),fontsize=12)
plt.title('Plot of 2D synthetic data and eigen directions',fontsize=15)
plt.show(False)
import math 
proj1=np.dot(d,np.transpose(vec[:,0])) 
angle=math.atan(vec[1,0]/vec[0,0])
plt.scatter(-1*proj1*math.cos(angle),
            -1*proj1*math.sin(angle),
            marker='x', color='r')
plt.show()

plt.scatter(d[:,0],d[:,1],marker='x')
plt.axis('equal')
plt.xticks(range(-15, 20, 5))
plt.yticks(range(-15, 20, 5))
plt.title('scatter plot',fontsize=15)
plt.xlabel('x1',fontsize=18)
plt.ylabel('y1',fontsize=18)


val,vec=np.linalg.eig(cov)
print('eigen values are:')
print(val[0],val[1])
print('Eigen vectors are:')
print((vec))


plt.quiver(0,0,vec[0,1],vec[1,1],scale=3)
plt.quiver(0,0,vec[0,0],vec[1,0],scale=6)
plt.xticks(range(-20, 20, 5),fontsize=12)
plt.yticks(range(-15, 20, 5),fontsize=12)
plt.title('Plot of 2D synthetic data and eigen directions',fontsize=15)


proj2=np.dot(d,np.transpose(vec[:,1])) 
angle=math.atan(vec[1,1]/vec[0,1])
plt.scatter(-1*proj2*math.cos(angle),
            -1*proj2*math.sin(angle),
            marker='x', color='r')
plt.show()    
    
D_cap1 = np.dot(proj1.reshape(1000,1), vec[:, 0].T.reshape(1, 2))
D_cap2 = np.dot(proj2.reshape(1000,1), vec[:, 1].T.reshape(1, 2))

print((((D_cap1 - d)**2).sum())/1000)
print((((D_cap2 - d)**2).sum())/1000)


#Question3
from sklearn.decomposition import PCA

#3a)
df.drop(['dates','stationid'],axis=1,inplace=True)
pca=PCA(n_components=2)#n_components specifies the no.of dimensions to be retained(l)
#The fit method learns some quantities from the data
pca.fit(df)
# The transform method applies the dimensionality reduction on df.
df_pca=pca.transform(df)
print("The variance of the projected data along the two directions is:" )
#The explained_variance_ method gives variance of each of the selected components which is equal 
#to n_components largest eigenvalues of the covariance matrix of df.
print(pca.explained_variance_)
covmat=df.cov()#calculating the covariance matrix of df
evalue,evector=np.linalg.eig(covmat)
print("The eigen values along the corresponding two directions are:" )
print(evalue[:2])#The 2 largest eigen values of covmat
plt.scatter(df_pca[:,0],df_pca[:,1])
plt.xlabel("Component1")
plt.ylabel("Component2")
plt.title("Scatterplot of reduced dimensional data")
plt.show()
print(evalue)
#3b)
evalue.sort()
plt.plot(evalue[::-1])
plt.ylabel("Eigan values")
plt.title("Plot of Eigen Values in descending order")
plt.show()

#3c)
#l=1
pca1=PCA(n_components=1)
df_pca1=pca1.fit_transform(df)
df_orig1=pca1.inverse_transform(df_pca1)
df_orig1=df_orig1.flatten()
#l=3
pca3=PCA(n_components=3)
df_pca3=pca3.fit_transform(df)
df_orig3=pca3.inverse_transform(df_pca3)
df_orig3=df_orig3.flatten()
#l=4
pca4=PCA(n_components=4)
df_pca4=pca4.fit_transform(df)
df_orig4=pca4.inverse_transform(df_pca4)
df_orig4=df_orig4.flatten()
#l=5
pca5=PCA(n_components=5)
df_pca5=pca5.fit_transform(df)
df_orig5=pca5.inverse_transform(df_pca5)
df_orig5=df_orig5.flatten()
#l=6
pca6=PCA(n_components=6)
df_pca6=pca6.fit_transform(df)
df_orig6=pca6.inverse_transform(df_pca6)
df_orig6=df_orig6.flatten()
#l=7
pca7=PCA(n_components=7)
df_pca7=pca7.fit_transform(df)
df_orig7=pca7.inverse_transform(df_pca7)
df_orig7=df_orig7.flatten()
#l=2
df_orig=pca.inverse_transform(df_pca)
df_orig=df_orig.flatten()

rmse=[]
df=np.array(df)
df=df.flatten()
r1=np.sqrt(((df - df_orig1) ** 2).mean())
rmse.append(round(r1,3))
r2=np.sqrt(((df - df_orig) ** 2).mean())
rmse.append(round(r2,3))
r3=np.sqrt(((df - df_orig3) ** 2).mean())
rmse.append(round(r3,3))
r4=np.sqrt(((df - df_orig4) ** 2).mean())
rmse.append(round(r4,3))
r5=np.sqrt(((df - df_orig5) ** 2).mean())
rmse.append(round(r5,3))
r6=np.sqrt(((df - df_orig6) ** 2).mean())
rmse.append(round(r6,3))
r7=np.sqrt(((df - df_orig7) ** 2).mean())
rmse.append(round(r7,3))

plt.bar([1,2,3,4,5,6,7],rmse)
plt.xlabel("Reduced Dimensions")
plt.ylabel("Reconstruction Error")
plt.title("Plot of Reconstruction Errors")
plt.show()
print(rmse)
  ## ################################################
  
##THANK YOU
  




