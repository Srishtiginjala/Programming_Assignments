#IC252LAB4
#Name         :Srishti Ginjala
#Rollno       :B19084

import random

#Question1
x1=0
x2=0;x12=0;z2=0;z12=0;z_cond=0;z_x1=z_x2=0
N=100000
for _ in range(N):
    X1=random.randint(0, 1)
    X2=random.randint(0, 1)
    Z1=X1+X2
    if(Z1==2):
        z2+=1
        if X1==1 and X2==1:
            z_cond+=1
        if X1==1:
            z_x1+=1
        if X2==1:
            z_x2+=1
    if(X1):
        x1+=1
    if(X2):
        x2+=1
    if X1==1 and X2==1:
        x12+=1
    if X1==1 and Z1==2:
        z12+=1
print("P(X1)=",x1/N)
print("P(X2)=",x2/N)
print("P(X1X2)=",x12/N)
print("P(X1)P(X2)=",x1*x2/N**2)
print("P(X1X2)=P(X1)P(X2). Thus X1 and X2 are independent. Also P(X1)=P(X2)\n")
print("P(Z1)=",z2/N)
print("P(X1Z1)=",z12/N)
print("P(X1)P(Z1)=",x1*z2/N**2)
print("P(X1Z1)!=P(X1)P(Z1). Thus X1 and Z1 are dependent.\n")
print("P(X1=1,X2=1/Z=1)",z_cond/N)
print("P(X1=1/Z1=1)=",z_x1/N)
print("P(X2=1/Z1=1)=",z_x2/N)
print("Hence, they are not conditionally independent")

#Question2
import numpy as np
import matplotlib.pyplot as plt
N=int(input())
p=float(input())
s=np.random.binomial(N,p,10000)
plt.hist(s)
plt.show()

#Question3
p=float(input())
l=[]
for _ in range(10000):
    count=0
    while(True):
        s=np.random.binomial(1,p)
        count+=1
        if(s):
            l.append(count)
            break;
plt.hist(l)
plt.show()

#THANK YOU


