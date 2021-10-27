#IC272LAB1
#Name         :Srishti Ginjala
#Rollno       :B19084
#Mobile Number:9440000900


import pandas as pd
import statistics as sc
import matplotlib.pyplot as plt

#Reading the file and storing all attributes
df=pd.read_csv("landslide_data3.csv")
temp=df["temperature"]
hum=df["humidity"]
pre=df["pressure"]
rain=df["rain"]
lavg=df["lightavgw/o0"] 
lmax=df["lightmax"]
mois=df["moisture"]

#Question1
#Function to find the Mean, median, mode, minimum, maximum and standard deviation
def props(l):
    mean=sc.mean(l)
    print("Mean:%0.5f"%mean)
    median=sc.median(l)
    print("Median:%0.5f"%median)
    mode=sc.mode(l)
    print("Mode:%0.5f"%mode)
    minimum=min(l)
    print("Minimum:%0.5f"%minimum)
    maximum=max(l)
    print("Maximum:%0.5f"%maximum)
    std=sc.stdev(l)
    print("Standard Deviation:%0.5f\n"%std)
print("Statistical parameters of Temperature:")
props(temp)
print("Statistical parameters of Humidity:")
props(hum)
print("Statistical parameters of Pressure:")
props(pre)
print("Statistical parameters of Rain:")
props(rain)
print("Statistical parameters of Light Average:")
props(lavg)
print("Statistical parameters of Maximum Light:")
props(lmax)
print("Statistical parameters of Moisture:")
props(mois)

#Question2
#2a)
#Function to find the scatter plot between ‘rain’ and each of the other attributes
def scatter(l,st):
    plt.scatter(rain,l)
    plt.ylabel(st,fontsize="15")
    plt.xlabel("rain",fontsize="15")
    plt.title("Scatter plot between rain and %s"%st,fontsize="20")
    plt.show()
scatter(temp,"temperature")
scatter(lmax,"Maximum light")
scatter(hum,"humidity")
scatter(pre,"pressure")
scatter(lavg,"lightavgw/o0")
scatter(mois,"moisture")

#2b)
#Function to find the scatter plot between ‘temperature’ and each of the other attributes
def scatter(l,st):
    plt.scatter(temp,l)
    plt.ylabel(st,fontsize="15")
    plt.xlabel("temperature",fontsize="15")
    plt.title("Scatter plot between temperature and %s"%st,fontsize="20")
    plt.show()
scatter(rain,"rain")
scatter(lmax,"Maximum light")
scatter(hum,"humidity")
scatter(pre,"pressure")
scatter(lavg,"lightavgw/o0")
scatter(mois,"moisture")

#Question3
#3a)
#We directly use the corr() function to find the value of correlation coefficient
print("The correlation coefficient of rain with temperature is %0.5f"%rain.corr(temp))
print("The correlation coefficient of rain with humidity is %0.5f"%rain.corr(hum))
print("The correlation coefficient of rain with Maximum Light is %0.5f"%rain.corr(lmax))
print("The correlation coefficient of rain with Average Light is %0.5f"%rain.corr(lavg))
print("The correlation coefficient of rain with pressure is %0.5f"%rain.corr(pre))
print("The correlation coefficient of rain with moisture is %0.5f"%rain.corr(mois))

#3b)
print("The correlation coefficient of temperature with rain is %0.5f"%temp.corr(rain))
print("The correlation coefficient of temperature with pressure is %0.5f"%temp.corr(pre))
print("The correlation coefficient of temperature with Maximum light is %0.5f"%temp.corr(lmax))
print("The correlation coefficient of temperature with humidity is %0.5f"%temp.corr(hum))
print("The correlation coefficient of temperature with Average light is %0.5f"%temp.corr(lavg))
print("The correlation coefficient of temperature with Moisture is %0.5f"%temp.corr(mois))

#Question4
plt.hist(mois,color='pink')
plt.title("Histogram of moisture",fontsize=15)
plt.show()

plt.hist(rain)
plt.title("Histogram of rain",fontsize=15)
plt.show()

#Question5
grouped=df.groupby('stationid')#returns a groupby object 
stations=(grouped.groups).keys()#grouped.groups returns a dictionary with stationid as keys and index values of its elements as values
for i in stations:    #iterating through all the stations
    df_rain=grouped.get_group(i) #retrieving the individual groups from the grouped object 
    rain_1=df_rain['rain']#pick the rain attribute from the data frame
    plt.hist(rain_1)#plot the histtogram of all values in that station
    plt.title('Histogram of rain in %s'%i,fontsize=15)
    plt.show()

#Question6
print("Boxplot of rain")
print(df.boxplot(column=['rain']))
#-->
#please run them separately as their ranges are quite different#
print("Boxplot of moisture")
print(df.boxplot(column=['moisture']))

#------------THE END---------------------------------------------------------------------------------------------
#--------------------------------------THANKYOU------------------------------------------------------------------------





