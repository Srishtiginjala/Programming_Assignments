#Question1

lx=[15,17,20,21,25]
ly=[9,13,16,18,21]
n=len(lx)
s=0
for i in lx:
    s+=i
Ex=s/n
s=0
for i in ly:
    s+=i
Ey=s/n
s=0
for i in range(n):
    s+=(lx[i]-Ex)*(ly[i]-Ey)
Cov=s/(n-1)
print("The covariance between X and Y is %0.2f"%Cov)

#Question2

sx=0;sy=0
for i in range(n):
    sx+=(lx[i]-Ex)**2
    sy+=(ly[i]-Ey)**2
Varx=sx/(n-1)
Vary=sy/(n-1)
cor=Cov/(Varx*Vary)**0.5
print("The correlation between X and Y is %0.2f"%cor)

#Question3

import pandas as pd
df=pd.read_csv("BNG-Device.csv")
def correlation(lx,ly):
    n=len(lx)
    s=0
    for i in range(n):
        if (lx.isnull()[i]==False):
            s+=lx[i]
    Ex=s/n
    s=0
    for i in range(n):
        if (ly.isnull()[i]==False):
            s+=ly[i]
    Ey=s/n
    s=0
    for i in range(n):
        if (ly.isnull()[i]==False) and (lx.isnull()[i]==False):
            s+=(lx[i]-Ex)*(ly[i]-Ey)
    Cov=s/(n-1)
    sx=0;sy=0
    for i in range(n):
        if (ly.isnull()[i]==False) and (lx.isnull()[i]==False):
            sx+=(lx[i]-Ex)**2
            sy+=(ly[i]-Ey)**2
    Varx=sx/(n-1)
    Vary=sy/(n-1)
    cor=Cov/(Varx*Vary)**0.5   
    return cor
def Relationship(value):
    if value==0:
        print("The relationship between them is None\n")
    elif (value>0 and value<=0.1) or (value<0 and value>=-0.1):
        print("The relationship between them is Weak\n")
    elif (value>0.1 and value<=0.3) or (value<-0.1 and value>=-0.3):
        print("The relationship between them is Moderate\n")
    elif (value>0.3 and value<=0.5) or (value<-0.3 and value>=-0.5):
        print("The relationship between them is Strong\n")
    elif (value>0.5 and value<1) or (value<-0.5 and value>-1):
        print("The relationship between them is Very Strong\n")
    elif value==1 or value==-1:
        print("The relationship between them is Perfect\n")

#3a)
lx=df["Active-Count"]
ly=df["CPU-Utilization"]
value=correlation(lx,ly)
print("The correlation between Active-Count and CPU-Utilization is %0.3f"%value)
Relationship(value)

#3b)
lx=df["Total-Memory-Usage"]
ly=df["CPU-Utilization"]
value=correlation(lx,ly)
print("The correlation between Total-Memory-Usage and CPU-Utilization is %0.3f"%value)
Relationship(value)

#3c)
lx=df["Average-Temperature"]
ly=df["CPU-Utilization"]
value=correlation(lx,ly)
print("The correlation between Average-Temperature and CPU-Utilization is %0.3f"%value)
Relationship(value)

#3d)
lx=df["Average-Temperature"]
ly=df["Active-Count"]
value=correlation(lx,ly)
print("The correlation between Average-Temperature and Active-Count is %0.3f"%value)
Relationship(value)

#3e)
lx=df["Average-Temperature"]
ly=df["Total-Memory-Usage"]
value=correlation(lx,ly)
print("The correlation between Average-Temperature and Total-Memory-Usage is %0.3f"%value)
Relationship(value)

    