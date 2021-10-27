#IC252_LAB10
#ROLLNO:B19084
#NAME:SRISHTI GINJALA

from scipy.stats import uniform
import random

#Question1
point=uniform.rvs(loc=-1,scale=2,size=(100,2))
ans=0
for i in range(len(point)):
    if(point[i][0]**2+point[i][1]**2<1):
        ans+=1
print("The value of pi obtained when n=100 is %0.3f\n"%(ans*4/100))

point=uniform.rvs(loc=-1,scale=2,size=(1000,2))
ans=0
for i in range(len(point)):
    if(point[i][0]**2+point[i][1]**2<1):
        ans+=1
print("The value of pi obtained when n=1000 is %0.3f\n"%(ans*4/1000))

point=uniform.rvs(loc=-1,scale=2,size=(10000,2))
ans=0
for i in range(len(point)):
    if(point[i][0]**2+point[i][1]**2<1):
        ans+=1
print("The value of pi obtained when n=10000 is %0.3f\n"%(ans*4/10000))
print("Hence we observe that as the no.of sample points increases ,the value of pi nears 3.14\n") 

#-----------------------------------------------------------------------------------------------------------------------------

#Question2
x=uniform.rvs(loc=0,scale=1,size=100)
y=uniform.rvs(loc=0,scale=2,size=100)
ans=0
for i in range(len(x)):
    if(y[i]*(1+x[i]**2)<2):
        ans+=1
print("The value of integral when n=100 is %0.3f\n"%(ans*2/100))

x=uniform.rvs(loc=0,scale=1,size=1000)
y=uniform.rvs(loc=0,scale=2,size=1000)
ans=0
for i in range(len(x)):
    if(y[i]*(1+x[i]**2)<2):
        ans+=1
print("The value of integral when n=1000 is %0.3f\n"%(ans*2/1000))

x=uniform.rvs(loc=0,scale=1,size=10000)
y=uniform.rvs(loc=0,scale=2,size=10000)
ans=0
for i in range(len(x)):
    if(y[i]*(1+x[i]**2)<2):
        ans+=1
print("The value of integral when n=10000 is %0.3f\n"%(ans*2/10000))
print("Hence the value of the integral nears 1.57(actual area) as sample points increases")

#----------------------------------------------------------------------------------------------------------------------

#Question3
ans=0
l=[1,2,3,4,5,6,7,8,9,10]
for j in range(1000):
    random.shuffle(l)
    flag=1
    for i in range(10):
        if(l[i]==i+1):
            flag=0
            break
    if(flag==1):
        ans+=1
print("Value of e obtained when n=1000 is %0.5f\n"%(1000/ans))

ans=0
l=[1,2,3,4,5,6,7,8,9,10]
for j in range(3000):
    random.shuffle(l)
    flag=1
    for i in range(10):
        if(l[i]==i+1):
            flag=0
            break
    if(flag==1):
        ans+=1
print("Value of e obtained when n=3000 is %0.5f\n"%(3000/ans))

ans=0
l=[1,2,3,4,5,6,7,8,9,10]
for j in range(5000):
    random.shuffle(l)
    flag=1
    for i in range(10):
        if(l[i]==i+1):
            flag=0
            break
    if(flag==1):
        ans+=1
print("Value of e obtained when n=5000 is %0.5f\n"%(5000/ans))

ans=0
l=[1,2,3,4,5,6,7,8,9,10]
for j in range(10000):
    random.shuffle(l)
    flag=1
    for i in range(10):
        if(l[i]==i+1):
            flag=0
            break
    if(flag==1):
        ans+=1
print("Value of e obtained when n=10000 is %0.5f\n"%(10000/ans))
print("Hence we observe that estimated value of e nears 2.71828 as n increases")

#################################################
##THANK YOU



    
    