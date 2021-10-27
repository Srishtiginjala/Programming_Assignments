#Question1

#1a)
import matplotlib.pyplot as plt
import numpy as np
X=np.arange(0,0.2,0.00000001)
Y=[57*(np.e)**(-57*x) for x in X]
plt.plot(X,Y)
plt.show()

#1b)
#1min=1/60 hours ,P(x<=1/60)=?
print("Probability of the wait time for the next Covid-19confirmed case to be less than or equal to 1 minute is %0.4f\n"%(1-np.e**(-57/60)))

#1c)
ans=np.e**(-57/60)-np.e**(-57*2/60)
print("Probability of the wait time for the next Covid-19confirmed case to be between 1 minute and 2 minutes is %0.4f\n"%ans)

#1d)
print("Probability of the wait time for the next Covid-19confirmed case to be more than 2 minutes is %0.4f\n"%(np.e**(-57*2/60)))

#1e)
#Here lambda is 57*2=114
ans=np.e**(-114/60)-np.e**(-114*2/60)
print("Probability of the wait time for the next Covid-19confirmed case to be between 1 minute and 2 minutes is %0.4f\n"%ans)


#Question2

import pandas as pd
df=pd.read_csv("IC252_Lab8.csv")

#2a)
b=df["Status"]
t=b.map({'Recovered':2,'Hospitalized':1,'Dead':3})
def correlation(lx,ly):
    n=len(lx)
    s=0
    for i in range(n):
        s+=(lx[i])
    ex=s/n
    s=0
    for i in range(n):
        s+=(ly[i])
    ey=s/n
    s=0
    for i in range(n):
        s+=((lx[i])-ex)*(ly[i]-ey)
    cov=s/(n-1)
    sx=0
    sy=0
    for i in range(n):
        sx+=((lx[i])-ex)**2
        sy+=(ly[i]-ey)**2
    varx=sx/(n-1)
    vary=sy/(n-1)
    cor=cov/(varx*vary)**0.5
    return cor

#2b)
lx=df["Population"]
cora=correlation(t,lx)
print("correlation between the Status and Population is %0.4f\n"%cora)
corb=correlation(t,df["SexRatio"])
print("correlation between the Status and SexRatio is %0.4f\n"%corb)
corc=correlation(t,df["Literacy"])
print("correlation between the Status and Literacy is %0.4f\n"%corc)
cord=correlation(t,df["Age"])
print("correlation between the Status and Age is %0.4f\n"%cord)
core=correlation(t,df["SmellTrend"])
print("correlation between the Status and SmallTrend is %0.4f\n"%core)
corf=correlation(t,df["Gender"])
print("correlation between the Status and Gender is %0.4f\n"%corf)

#2c)
print("Hence the order of correlation to status is Age,Literacy,Gender,SmallTrend,Population,SexRatio.We can see that Age strongly correlates to status")