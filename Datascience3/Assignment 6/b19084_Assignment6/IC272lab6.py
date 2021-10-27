#IC272LAB6
#Name         :Srishti Ginjala
#Rollno       :B19084
#Mobile Number:9440000900

import matplotlib.pyplot as plt
import pandas as pd

#Question1
df=pd.read_csv(r"C:\Users\srish\Downloads\datasetA6_HP.csv")
orig=df["HP"]
#a)
plt.plot(df.index,df["HP"])
plt.title("Power consumed (in MW) vs. days")
plt.xlabel("Days")
plt.ylabel("Power(in MW)")
plt.show()
#b)
lag_one=df["HP"].shift(1)
print("The correlation coefficient of rain with temperature is %0.5f"%orig.corr(lag_one))
#c)
plt.scatter(orig,lag_one)
plt.title("Scatter plot one day lagged sequence vs. given time sequence")
plt.xlabel("Original Consumption")
plt.ylabel("1-day lag Consumption")
plt.show()
print("Yes,it does match with the calculated coefficient")
#d)
lagvalues=[1,2,3,4,5,6,7]
cor=[]
for i in lagvalues:
    lag_i=orig.shift(i)
    cor.append(orig.corr(lag_i))
plt.plot(lagvalues,cor)
plt.title("Correlation coefficient vs. lags in given sequence")
plt.xlabel("Lagvalues")
plt.ylabel("Correlation coefficient")
plt.show()
#e)
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(orig, lags=lagvalues)
plt.title("Correlation coefficient vs. lags in given sequence generated using 'plot_acf' function")
plt.xlabel("Lagvalues")
plt.ylabel("Correlation coefficient")
plt.show()

#Question2
X=df.values
test=X[len(X)-250:]
pred=test[:,1]
act=lag_one.values
act1=act[len(act)-250:]
from sklearn.metrics import mean_squared_error as msr
import math
rmse=math.sqrt(msr(act1,pred))
print("RMSE value is %0.3f"%rmse)

#3a)
from statsmodels.tsa.ar_model import AutoReg

X =df['HP'].values
train, test = X[0:len(X)-250], X[len(X)-250:]
# train autoregression
model = AutoReg(train, lags=5)
model_fit = model.fit()
print('Coefficients:' , model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
x=[]
y=[]
for i in range(len(predictions)):
    y.append(predictions[i])
    x.append(test[i])
rmse = (msr(test, predictions))**.5
print('Test RMSE:', rmse)
plt.plot(test)
plt.plot(predictions)
plt.title('orginal data vs predicted data',fontsize=18)
plt.xlabel('Index of day', fontsize=18,c='b')
plt.ylabel('Power Consumed ', fontsize=18,c='g')
plt.show()
#3b)
l=[1,5,10,15,25]
for i in l:
    model = AutoReg(train, lags=i)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    rmse = (msr(test, predictions))**.5
    print('Test RMSE for lag '+ str(i) + ' is', rmse)
#3c)
i=1
s=pd.Series(list(df['HP']))
while(s.autocorr(i)>2/(250-i)**.5):
     i=i+1
#3d)
model2 = AutoReg(train, lags=5)
model_fit2 = model2.fit()
pred2 = model_fit2.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

print('RMSE for test where Heuristic optimal lags of ' +str(5))
print('%.3f'%msr(test, pred2, squared=False))
print('\n')
print('Optimal lags of lags(p) is:' + str(25))
print('RMSE of the Optimal lags:' +str(4.515))
print('Heuristic value of optimal lags:'+ str(5))
print('RMSE of Heuristic optimal lags:' +str(4.538))

#Question3
from statsmodels.tsa.ar_model import AutoReg
# load dataset
series= pd.read_csv(r"C:\Users\srish\Downloads\datasetA6_HP.csv")
# split dataset
X = series['HP'].values
train, test = X[1:len(X)-250], X[len(X)-250:]
# train autoregression
model = AutoReg(train, lags=5)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%0.3f, expected=%0.3f' % (predictions[i], test[i]))
rmse = math.sqrt(msr(test, predictions))
print('Test RMSE: %.3f' % rmse)
