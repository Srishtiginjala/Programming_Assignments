#IC272LAB5
#Name         :Srishti Ginjala
#Rollno       :B19084
#Mobile Number:9440000900


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

#Part A
#Question1

df=pd.read_csv("seismic_bumps1.csv")
df=df.drop(['nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89'],axis=1)
grouped=df.groupby('class')
df_1=grouped.get_group(1)
df_0=grouped.get_group(0)

y1=df_1['class']
y0=df_0['class']
[df_train1,df_test1,df_label_train1,df_label_test1] =train_test_split(df_1,y1,test_size=0.3,random_state=42,shuffle=True)
[df_train0,df_test0,df_label_train0,df_label_test0] =train_test_split(df_0,y0,test_size=0.3,random_state=42,shuffle=True)
test=pd.concat([df_test0,df_test1])
train=pd.concat([df_train0,df_train1]) 

qs = [2,4,8,16]
acc = []#stores the accuracy
clss = [0,1]
p = [len(train[train['class']==c])/len(train) for c in clss]##stores the prior probabilities
test1 = test.drop('class', axis=1)

for q in qs:
    p1 = np.zeros((len(test),2))
    pred = []
    for c in clss:
        x = train[train['class'] == c].copy()
        x = x.drop('class', axis=1)
        GMM = GaussianMixture(n_components = q, covariance_type='full', random_state=42)
        GMM.fit(x)
        sc = p[c] * np.exp(GMM.score_samples(test1))
        # sc = sc.reshape(sc.shape[0], 1)
        p1[:, c]=sc
    for i in range(p1.shape[0]):
        pred.append(p1[i].argmax())
    acc.append(accuracy_score(test['class'], pred))
    print(confusion_matrix(test['class'], pred))
    print(acc[-1])

#Question2
names = ['KNN with Unnormalized Data', 'KNN with Normalized Data', 'Bayes Classifier (Unimodal)', 'Bayes Classifier (GMM)']
accs = pd.DataFrame([0.927835, 0.926546, 0.889175, max(acc)], index=names, columns=['Accuracy'])
print(accs)

#Part B-----------------------------------------------------------------------------------------------------------
#Question1
df = pd.read_csv('atmosphere_data.csv')
train, test = train_test_split(df, test_size=0.3, random_state=42)
train.to_csv('atmosphere-train.csv')
test.to_csv('atmosphere-test.csv')

x_train = np.array(train['pressure']).reshape(-1,1)
y_train = np.array(train['temperature'])
x_test = np.array(test['pressure']).reshape(-1,1)
y_test = np.array(test['temperature'])

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Question1 (a)
fig,ax=plt.subplots()
ax.scatter(x_train, y_train, c='r')
ax.plot(np.linspace(min(x_train), max(x_train), 1000), [regressor.coef_*x + regressor.intercept_ for x in np.linspace(min(x_train), max(x_train), 1000)], label='Best Fit Line')
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title('Linear Regression')
plt.show()

#Question1 (b)
train_pred = regressor.predict(x_train)
print('Train RMSE :%0.3f'%mean_squared_error(train_pred,y_train)**.5)

#Question1 (c)
test_pred = regressor.predict(x_test)
print('Test RMSE :%0.3f'%mean_squared_error(test_pred, y_test)**.5)

#Question1 (d)
plt.scatter(y_test, test_pred)
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.show()

#Question2 (a)
P = [2,3,4,5]
train_acc=[]
test_acc=[]
best_coeffs=[]
for p in P:
    polynomial_features = PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    y_pred = regressor.predict(x_poly)
    print('p = %d'%p)
    train_acc.append(mean_squared_error(y_train, y_pred)**.5)
    print('Train RMSE :%0.3f'%train_acc[-1])
    
    x_test_poly = polynomial_features.fit_transform(x_test)
    y_test_pred = regressor.predict(x_test_poly)
    test_acc.append(mean_squared_error(y_test, y_test_pred)**.5)
    print('Test RMSE :%0.3f'%test_acc[-1])
    print()
    if p==2:
        best_coeff=regressor.coef_
        best_coeff[0]+=regressor.intercept_
    else:
        if test_acc[-2]>test_acc[-1]:
            best_coeff=regressor.coef_
            best_coeff[0]+=regressor.intercept_

plt.bar(range(2,len(train_acc)+2), train_acc)
plt.title('Train RMSE')
plt.xlabel('Degree of Polynomial')
plt.ylabel('RMSE')
plt.show()

#Question2 (b)    
plt.bar(range(2,len(test_acc)+2), test_acc)
plt.title('Test RMSE')
plt.xlabel('Degree of Polynomial')
plt.ylabel('RMSE')
plt.show()

#Question2 (c)
plot_x = np.linspace(min(x_train), max(x_train), 1000)
plot_y = [np.polyval(best_coeff[::-1], x) for x in plot_x]
plt.scatter(x_train, y_train, c='r')
plt.plot(plot_x, plot_y)
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.title('Polynomial Regression')
plt.show()

#Question2 (d)
y_test_pred = [np.polyval(best_coeff[::-1], x) for x in x_test]
plt.scatter(y_test, y_test_pred)
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.show()

## ################################################
  
##THANK YOU