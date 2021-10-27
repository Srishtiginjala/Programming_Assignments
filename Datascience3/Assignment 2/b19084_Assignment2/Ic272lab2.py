#IC272LAB2
#Name         :Srishti Ginjala
#Rollno       :B19084
#Mobile Number:9440000900


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Question1
df=pd.read_csv("pima_indians_diabetes_miss.csv")
ly=df.isnull().sum()
#The df.isnull func returns true if the value is missing(NaN) else false.
#The sum func returns a series counting all the true values in respective columns
lx=df.columns #lx is the list of column names in given df
plt.plot(lx,ly)
plt.xlabel("attributes")
plt.ylabel("no.of missing values")
plt.show()

#Question2
#a)
l=[]
for i in range(len(df.index)):
    if(df.loc[i].isnull().sum() >=3):
        l.append(i)
print("\nThe row numbers of the deleted tuples are:")
print(l)
print("No.of tuples deleted are %d\n"%len(l))

#b)
ln=[]
temp=df.isnull()
for i in range(len(df.index)):
    if(temp["class"][i]):
        ln.append(i)
print("The row numbers of the tuples having missing value in the target (class) attribute are :")
print(ln)
print("No.of tuples deleted are %d\n"%len(ln))
df_2=df.drop(l+ln)

#Question3
print("The no.of missing values in attributes are")
print(df_2.isnull().sum())
print("The total number of missing values in the file are %d\n"%df_2.isnull().sum().sum())

#Question4
#a) both 1) and 2) are done together
errors_a=[]
df_mean=df_2.fillna(df.mean(skipna=True))
df_orig=pd.read_csv("pima_indians_diabetes_original.csv")
doop=df_2.isna()
indexx=list(doop.index)
for i in lx:
    print("Replaced mean of %s:%0.3f "%(i,df_mean[i].mean()))
    print("Original mean of %s:%0.3f "%(i,df_orig[i].mean()))
    print("Replaced  mode of %s is\n"%i,df_mean[i].mode())
    print("Original mode of %s is\n"%i,df_orig[i].mode())
    print("Replaced median of %s:%0.3f "%(i,df_mean[i].median()))
    print("Original median of %s:%0.3f "%(i,df_orig[i].median()))
    print("Replaced standard deviation of %s:%0.3f "%(i,df_mean[i].std()))
    print("Original standard deviation of %s:%0.3f "%(i,df_orig[i].std()))
    l=0
    m=0
    for j in indexx:
        if(doop[i][j]==True):
            l=l+(df_mean[i][j]-df_orig[i][j])**2
            m=m+1
    if(m==0):
        errors_a.append(0)     
    else:    
        errors_a.append((l/m)**0.5) 
print("The RMSE values for replaced mean are:\n",errors_a)
plt.plot(lx,errors_a)
plt.title("root mean square errors")
plt.xlabel("attributes")
plt.ylabel("rmse")
plt.show()

#b)
errors_b=[]
df_int=df_2.interpolate(limit_direction='both')#df.interpolate() func is used to fill NA values in the dataframe 
for i in lx:
    print("Replaced mean of %s:%0.3f "%(i,df_int[i].mean()))
    print("Original mean of %s:%0.3f "%(i,df_orig[i].mean()))
    print("Replaced  mode of %s is\n"%i,df_int[i].mode())
    print("Original mode of %s is\n"%i,df_orig[i].mode())
    print("Replaced median of %s:%0.3f "%(i,df_int[i].median()))
    print("Original median of %s:%0.3f "%(i,df_orig[i].median()))
    print("Replaced standard deviation of %s:%0.3f "%(i,df_int[i].std()))
    print("Original standard deviation of %s:%0.3f "%(i,df_orig[i].std()))
    l=0
    m=0
    for j in indexx:
        if(doop[i][j]==True):
            l=l+(df_int[i][j]-df_orig[i][j])**2
            m=m+1
    if(m==0):
        errors_b.append(0)     
    else:    
        errors_b.append((l/m)**0.5)
print("The RMSE values for interpolated mean are:\n",errors_b)
plt.plot(lx,errors_b)
plt.title("root mean square errors")
plt.xlabel("attributes")
plt.ylabel("rmse")
plt.show()

#Question5
df_2=df_2.interpolate(limit_direction='both')
lage=list(df_2["Age"])
lbmi=list(df_2["BMI"])
q1=np.percentile(lage,25)
q3=np.percentile(lage,75)

outliers_age=[]
for i in df_2["Age"]:
    if(i>(2.5*q3-1.5*q1) or i<(2.5*q1-1.5*q3)):
        outliers_age.append(i)
print("\nThe outliers in Age are",outliers_age)

q1=np.percentile(lbmi,25)
q3=np.percentile(lbmi,75)
outliers_BMI=[]
for i in df_2["BMI"]:
    if(i>(2.5*q3-1.5*q1) or i<(2.5*q1-1.5*q3)):
        outliers_BMI.append(i)
print("\nThe outliers in BMI are",outliers_BMI)
df_2.boxplot(column=['Age'])
df_2.boxplot(column=['BMI'])
#b)
medage=df_2['Age'].median()
for i in range(len(lage)):
    if(lage[i]>(2.5*q3-1.5*q1) or lage[i]<(2.5*q1-1.5*q3)):
        df_2['Age'][i]=np.nan

medbmi=df_2['BMI'].median()
for i in range(len(lbmi)):
    if(lbmi[i]>(2.5*q3-1.5*q1) or lbmi[i]<(2.5*q1-1.5*q3)):
        df_2['BMI'][i]=np.nan
df_2['Age']=df_2['Age'].fillna(medage)
df_2['BMI']=df_2['BMI'].fillna(medbmi)

#run both of them separately
df_2.boxplot(column=['Age'])
df_2.boxplot(column=['BMI'])

#We get outliers because they have been replaced by the median 

