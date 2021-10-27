#IC252_LAB9
#ROLLNO:B19084
#NAME:SRISHTI GINJALA

from statistics import mean
import matplotlib.pyplot as plt
#Question1
lx=[10, 100, 500, 1000, 5000, 10000, 50000]
#1a)
from scipy.stats import expon
ly=[]
data_10=expon.rvs(scale=1,loc=0,size=10)
mean_10=mean(data_10)
ly.append(mean_10)

data_100=expon.rvs(scale=1,loc=0,size=100)
mean_100=mean(data_100)
ly.append(mean_100)


data_500=expon.rvs(scale=1,loc=0,size=500)
mean_500=mean(data_500)
ly.append(mean_500)


data_1000=expon.rvs(scale=1,loc=0,size=1000)
mean_1000=mean(data_1000)
ly.append(mean_1000)


data_5000=expon.rvs(scale=1,loc=0,size=5000)
mean_5000=mean(data_5000)
ly.append(mean_5000)


data_10000=expon.rvs(scale=1,loc=0,size=10000)
mean_10000=mean(data_10000)
ly.append(mean_10000)

data_50000=expon.rvs(scale=1,loc=0,size=50000)
mean_50000=mean(data_50000)
ly.append(mean_50000)

plt.scatter(lx,ly)
plt.plot(lx,[1,1,1,1,1,1,1],color='magenta',linestyle='dashed',label="E(x)=1")
plt.legend()
plt.title("Exponential Distribution")
plt.xlabel("Sample size")
plt.ylabel("sample Mean")
plt.show()


#1b)
from scipy.stats import uniform
ly=[]
data_10=uniform.rvs(scale=1,loc=1,size=10)
mean_10=mean(data_10)
ly.append(mean_10)

data_100=uniform.rvs(scale=1,loc=1,size=100)
mean_100=mean(data_100)
ly.append(mean_100)


data_500=uniform.rvs(scale=1,loc=1,size=500)
mean_500=mean(data_500)
ly.append(mean_500)


data_1000=uniform.rvs(scale=1,loc=1,size=1000)
mean_1000=mean(data_1000)
ly.append(mean_1000)


data_5000=uniform.rvs(scale=1,loc=1,size=5000)
mean_5000=mean(data_5000)
ly.append(mean_5000)


data_10000=uniform.rvs(scale=1,loc=1,size=10000)
mean_10000=mean(data_10000)
ly.append(mean_10000)

data_50000=uniform.rvs(scale=1,loc=1,size=50000)
mean_50000=mean(data_50000)
ly.append(mean_50000)

plt.scatter(lx,ly)
plt.plot(lx,[1.5,1.5,1.5,1.5,1.5,1.5,1.5],color='magenta',linestyle='dashed',label="E(x)=1.5")
plt.legend()
plt.title("Uniform Distribution")
plt.xlabel("Sample size")
plt.ylabel("sample Mean")
plt.show()


#1c)
from scipy.stats import bernoulli
ly=[]
data_10=bernoulli.rvs(p=0.2,size=10)
mean_10=(sum(data_10)/10)
ly.append(mean_10)

data_100=bernoulli.rvs(p=0.2,size=100)
mean_100=(sum(data_100)/100)
ly.append(mean_100)


data_500=bernoulli.rvs(p=0.2,size=500)
mean_500=(sum(data_500)/500)
ly.append(mean_500)


data_1000=bernoulli.rvs(p=0.2,size=1000)
mean_1000=(sum(data_1000)/1000)
ly.append(mean_1000)


data_5000=bernoulli.rvs(p=0.2,size=5000)
mean_5000=(sum(data_5000)/5000)
ly.append(mean_5000)


data_10000=bernoulli.rvs(p=0.2,size=10000)
mean_10000=(sum(data_10000)/10000)
ly.append(mean_10000)

data_50000=bernoulli.rvs(p=0.2,size=50000)
mean_50000=(sum(data_50000)/50000)
ly.append(mean_50000)

plt.scatter(lx,ly)
plt.plot(lx,[0.2,0.2,0.2,0.2,0.2,0.2,0.2],color='magenta',linestyle='dashed',label="E(x)=0.2")
plt.legend()
plt.title("Bernoulli Distribution")
plt.xlabel("Sample size")
plt.ylabel("sample Mean")
plt.show()
print("From the above plots we can conclude that as the sample size increases,sample mean is approximately the Expectation of the respective distribution.Hence WLLN is verified\n")

#-----------------------------------------------------------------------------------------------------------------------

#Question2
import numpy as np
from scipy.stats import norm
def plotexp(l,n):
    freq,bins=np.histogram(l,bins=100)
    bin_centers=0.5*(bins[1:]+bins[:-1])
    probs=freq/10000
    pdf=norm.pdf(bin_centers,1,1/(n**0.5))
    plt.title("n=%d"%n)
    plt.bar(bin_centers,probs,1/25,color='pink')
    plt.show()
    plt.title("n=%d"%n)
    plt.plot(bin_centers,pdf,label="PDF",color='black')
    plt.legend()
    plt.show()
#2a)
exp=np.random.exponential(scale=1,size=(10000,1))
l=[]
for i in range(10000):
    l.append(mean(exp[i]))
plotexp(l,1)
#n=2
exp=np.random.exponential(scale=1,size=(10000,2))
l=[]
for i in range(10000):
    l.append(mean(exp[i]))
plotexp(l,2)

#n=4
exp=np.random.exponential(scale=1,size=(10000,4))
l=[]
for i in range(10000):
    l.append(mean(exp[i]))
plotexp(l,4)

#n=8
exp=np.random.exponential(scale=1,size=(10000,8))
l=[]
for i in range(10000):
    l.append(mean(exp[i]))
plotexp(l,8)

#n=16
exp=np.random.exponential(scale=1,size=(10000,16))
l=[]
for i in range(10000):
    l.append(mean(exp[i]))
plotexp(l,16)

#n=32
exp=np.random.exponential(scale=1,size=(10000,32))
l=[]
for i in range(10000):
    l.append(mean(exp[i]))
plotexp(l,32)
print("Hence as the value of n increases,the distribution nears normal distribution\n")

def plotuni(l,n):
    freq,bins=np.histogram(l,bins=100)
    bin_centers=0.5*(bins[1:]+bins[:-1])
    probs=freq/10000
    pdf=norm.pdf(bin_centers,1.5,1/((12*n)**0.5))
    plt.title("n=%d"%n)
    plt.bar(bin_centers,probs,1/25,color='pink')
    plt.show()
    plt.title("n=%d"%n)
    plt.plot(bin_centers,pdf,label="PDF",color='black')
    plt.legend()
    plt.show()
#2b)
uni=np.random.uniform(low=1,high=2,size=(10000,1))
l=[]
for i in range(10000):
    l.append(mean(uni[i]))
plotuni(l,1)

#n=2
uni=np.random.uniform(low=1,high=2,size=(10000,2))
l=[]
for i in range(10000):
    l.append(mean(uni[i]))
plotuni(l,2)
 
#n=4
uni=np.random.uniform(low=1,high=2,size=(10000,4))
l=[]
for i in range(10000):
    l.append(mean(uni[i]))
plotuni(l,4)

#n=8
uni=np.random.uniform(low=1,high=2,size=(10000,8))
l=[]
for i in range(10000):
    l.append(mean(uni[i]))
plotuni(l,8)

#n=16
uni=np.random.uniform(low=1,high=2,size=(10000,16))
l=[]
for i in range(10000):
    l.append(mean(uni[i]))
plotuni(l,16)

#n=32
uni=np.random.uniform(low=1,high=2,size=(10000,32))
l=[]
for i in range(10000):
    l.append(mean(uni[i]))
plotuni(l,32)
print("Hence as the value of n increases,the distribution nears normal distribution\n")

def plotber(l,n):
    freq,bins=np.histogram(l,bins=100)
    bin_centers=0.5*(bins[1:]+bins[:-1])
    probs=freq/10000
    pdf=norm.pdf(bin_centers,0.2,0.4/(n**0.5))
    plt.title("n=%d"%n)
    plt.bar(bin_centers,probs,1/25,color='pink')
    plt.show()
    plt.title("n=%d"%n)
    plt.plot(bin_centers,pdf,label="PDF",color='black')
    plt.legend()
    plt.show()
#2c)
ber=np.random.binomial(n=1,p=0.2,size=(10000,1))
l=[]
for i in range(10000):
    l.append(sum(ber[i]))
plotber(l,1) 

#n=2
ber=np.random.binomial(n=1,p=0.2,size=(10000,2))
l=[]
for i in range(10000):
    l.append(sum(ber[i])/2)
plotber(l,2) 

#n=4
ber=np.random.binomial(n=1,p=0.2,size=(10000,4))
l=[]
for i in range(10000):
    l.append(sum(ber[i])/4)
plotber(l,4) 

#n=8
ber=np.random.binomial(n=1,p=0.2,size=(10000,8))
l=[]
for i in range(10000):
    l.append(sum(ber[i])/8)
plotber(l,8) 
 
#n=16
ber=np.random.binomial(n=1,p=0.2,size=(10000,16))
l=[]
for i in range(10000):
    l.append(sum(ber[i])/16)
plotber(l,16) 

#n=32
ber=np.random.binomial(n=1,p=0.2,size=(10000,32))
l=[]
for i in range(10000):
    l.append(sum(ber[i])/32)
plotber(l,32) 
 
print("Hence as the value of n increases,the distribution nears normal distribution.Hence CLT is verified.\n")
#################################################
##THANK YOU



