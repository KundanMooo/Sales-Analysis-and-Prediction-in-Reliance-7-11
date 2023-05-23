#Importing libraries 
import math                
import numpy as np
import pandas as pd
import seaborn as sns                                 


from statsmodels.formula import api              
from sklearn.linear_model import LinearRegression 
from sklearn.feature_selection import RFE          
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split 

from IPython.display import display


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 

import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings('ignore')


#data Ingestion 
df=pd.read_csv('D:/DS/resume projects/7-eleven/seven_eleven_sales.csv')

#Feature Engineering
df['Date']=pd.to_datetime(df['Date'])
df['month']=df['Date'].dt.month
df['Year']=df['Date'].dt.year
df['Week']=df['Date'].dt.weekday
df=df.drop(['Date'],axis=1)
df.info()


#Separating categorical and numerical columns
catcol=[]
numcol=[]

s=df.nunique().sort_values()

for i in s.index:
  if s[i]<=45: catcol.append(i)
  else: numcol.append(i)
  

#EDA
#1. categorical columns

fig,axes=plt.subplots(1,len(catcol),figsize=(15,4))
for kk,i in enumerate(catcol):
    df.groupby(i)['Weekly_Sales'].sum().plot(kind='bar',ax=axes[kk])
    axes[kk].set_title('Sales by '+ i)
    
#2. numerical columns

fig,axes=plt.subplots(1,len(numcol),figsize=(15,4))
for kk,i in enumerate(numcol):
    sns.histplot(df[i],ax=axes[kk],kde=True)
    axes[kk].set_title('Sales by '+ i)
#2.5 outliers     
plt.figure(figsize=(15,4))
for k,i in enumerate(numcol):
  plt.subplot(1,5,k+1)
  sns.boxplot(df[i])
  plt.title(i)
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()

#3. Weekly sales analysis
plt.figure(figsize=[10,12])    
plt.subplot(221)
df.groupby('Holiday_Flag')['Weekly_Sales'].sum().plot(kind='bar')
plt.title('Sum of Weekly sales by Holiday')

plt.subplot(222)
df.groupby('Holiday_Flag')['Weekly_Sales'].mean().plot(kind='bar')
plt.title('Average of Weekly sales by Holiday')

plt.subplot(614)
df.groupby('month')['Weekly_Sales'].sum().plot(kind='bar')
plt.title('Sum of Weekly sales by month')

plt.subplot(615)
df.groupby('month')['Weekly_Sales'].mean().plot(kind='bar')
plt.title('Average of Weekly sales by month')

plt.subplot(616)
df.groupby('Store')['Weekly_Sales'].sum().plot(kind='bar')
plt.title('Sum of Weekly sales by Store')


#remove outlier
for i in numcol:
  q1=dfumm[i].quantile(0.25)
  q3=dfumm[i].quantile(0.75)
  iqr=q3-q1
  h=q3+(1.5*iqr)
  l=q1-(1.5*iqr)
  dfumm=dfumm[(dfumm[i]>=l) & (dfumm[i]<=h)]
print(plt.pie([len(dfumm),len(df)-len(dfumm)],labels=['Retained','Removed Outlier'],radius=1,autopct='%1.2f%%'))


#split 80/20
y=df['Weekly_Sales']
x=[i for i in df.columns if i not in 'Weekly_Sales']
x=dfumm.drop(['Weekly_Sales'],axis=1)
y=dfumm['Weekly_Sales']
xn,xs,yn,ys=train_test_split(x,y,train_size=0.8)

# one hot ecoding on categorical column
dfht=pd.DataFrame({})
for i in catcol:
  dfht=pd.concat([dfht,pd.get_dummies(df[i],drop_first=False,prefix=str(i))],axis=1)
dfumm=pd.concat((df,dfht),axis=1)

#Standardization of test and train
std=StandardScaler()
xstdn=std.fit_transform(xn)
xstds=std.transform(xs)
xstdn=pd.DataFrame(xstdn,columns=x.columns)
xstds=pd.DataFrame(xstds,columns=x.columns)


#rfe
ln=[]
ls=[]
col=len(x.columns)
for i in range(col):
    lr=LinearRegression()
    rfe=RFE(lr,n_features_to_select=xstdn.shape[1]-i)
    rfe=rfe.fit(xstdn,yn)
    
    LR=LinearRegression()
    cn=xstdn.loc[:,rfe.support_]
    cs=xstds.loc[:,rfe.support_]
    LR.fit(cn,yn)
    ypn=LR.predict(cn)
    yps=LR.predict(cs)
    
    ls.append(r2_score(ys,yps))
    ln.append(r2_score(yn,ypn))

plt.plot(ls,label='Test')
plt.plot(ln,label='Train')
plt.title('R2 curve')
plt.legend()
plt.grid()
plt.show()


#Based on the R2 curve removing 20 features 
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=xstdn.shape[1]-20)
rfe=rfe.fit(xstdn,yn)
    
LR=LinearRegression()
cn=xstdn.loc[:,rfe.support_]
cs=xstds.loc[:,rfe.support_]
LR.fit(cn,yn)
ypn=LR.predict(cn)
yps=LR.predict(cs)
    
print(r2_score(ys,yps))
print(r2_score(yn,ypn))

xrfen=xstdn.loc[:,rfe.support_]
xrfes=xstds.loc[:,rfe.support_]


# Multiple Linear Regression ( on reduced data after RFE )
mlr=LinearRegression().fit(xrfen,yn)
print('coefficents:-\n',mlr.coef_)
print('intercept:-\n',mlr.intercept_)


#test vs prediction
plt.scatter(ys,yps)
plt.plot([min(ys),max(ys)],[min(ys),max(ys)],'r--',label='Test line')
plt.plot([min(yps),max(yps)],[min(yps),max(yps)],'y--',label='Predict Line')
plt.xlabel('Y-Test')
plt.ylabel('Y-Predict')
plt.title('Test Vs Predict')
plt.legend()

print('train r2_score',r2_score(yn,ypn))
print('test r2_score',r2_score(ys,yps))