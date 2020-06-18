import pandas as pd
import datetime
import numpy as np
from array import *
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
from sklearn import svm
from xgboost import XGBClassifier as Xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from FunctionList import buyjudge,stochastic_oscillator,plot_precision_recall_vs_threshold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier   
matplotlib.style.use('ggplot')

method_name = [{
                # 'Random Forrest':RandomForestClassifier(),
                # 'Random Forrest30':RandomForestClassifier(oob_score=True, random_state=30),
                # 'Bayes(smo=1e-01)':GaussianNB(var_smoothing=1e-01),
                # 'Bayes(smo=0.5)':GaussianNB(var_smoothing=0.5),
                # 'Bayes(smo=1)':GaussianNB(var_smoothing=1),
                # 'Bayes(smo=2)':GaussianNB(var_smoothing=2),
                # 'SVC(C=1)':svm.SVC(probability=True),
                # 'SVC(linear, C=1)':svm.SVC(kernel='linear', C=1,probability=True),
                # 'SVC(poly, C=1)':svm.SVC(kernel='poly',probability=True),
                # 'XGBT(λ=1)':Xgb(reg_lambda=1),#Result of parameter tunning in XGBPara.py
                # 'XGBT(λ=1.2)':Xgb(reg_lambda=1.2)
                }]
method_list=pd.DataFrame(method_name)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])
start = datetime.datetime(2005,1,1)
end = datetime.date.today()
df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)
df_VIX = web.DataReader("^VIX", 'yahoo', start,end)
testduration=-180

stock='MSFT' #Load ticker data'MSFT','AAPL','AMZN','GOOG','FB','JNJ','V','PG','JPM','UNH','MA','INTC','VZ','HD','T','PFE','MRK','PEP']
df=web.DataReader(stock, 'yahoo', start, end).drop(columns=['Adj Close'])
rawdata=df
#Add features in
df['MAVOL200'] = df['Volume']/df['Volume'].rolling(200).mean()
df['MAVOL20'] = df['Volume']/df['Volume'].rolling(20).mean()
df['MAVOL10'] = df['Volume']/df['Volume'].rolling(10).mean()
df['MAVOL5'] = df['Volume']/df['Volume'].rolling(5).mean() 
df['SP500'] = df_SP500['Close']
df['SP500_ROC'] = 100*df['SP500'].diff(1)/df['SP500'].shift(1)
df['VIX'] = df_VIX['Close']
df['VIXMA5'] = df['VIX']/df['VIX'].rolling(5).mean()
df['VIXMA10'] = df['VIX']/df['VIX'].rolling(10).mean()
df['VIX_ROC'] = 100*df['VIX'].diff(1)/df['VIX'].shift(1)
df['Close_ROC'] = 100*df['Close'].diff(1)/df['Close'].shift(1)   
stochastic_oscillator(df)
df['Intersection'] = 0
df.loc[(df['K']>df['D']) & (df['K_prev']<df['D_prev']) & (df['D']<=80)  & (df['D_diff']>0),'Intersection'] = 1# Intersections: K go exceeds D   
df['# Inter 10-day'] = df['Intersection'].rolling(14).sum()# number of intersections during past 10 days    
df['Close/MA10']= df['Close']/df['Close'].rolling(10).mean()# df['MA10']= df['Close'].rolling(10) Moving Average of the past 10 days
df['Close/MA20']= df['Close']/df['Close'].rolling(20).mean()
df['Close/MA50']= df['Close']/df['Close'].rolling(50).mean()
df['Close/MA100']= df['Close']/df['Close'].rolling(100).mean()
df['Close/MA200']= df['Close']/df['Close'].rolling(200).mean()
df['VAR5']= df['Close_ROC'].rolling(5).std()
df['VAR10']= df['Close_ROC'].rolling(10).std()
buyjudge(df)
df.dropna(axis=0, how='any', inplace=True)#Get rid of rows with NA value

#Retrive X and y 
X=df.loc[:,['# Inter 10-day','Intersection','MAVOL200','MAVOL20','MAVOL10','MAVOL5','SP500_ROC',
            'VIX_ROC','VIXMA5','VIXMA10','Close_ROC','rsv','K','D','J',
            'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','Close/MA10',
            'Close/MA20','Close/MA50','Close/MA100','Close/MA200','VAR5','VAR10']]

X = preprocessing.MinMaxScaler().fit_transform(X) 
y=df.loc[:,'Good Buy Point?']

# Split train set and test set
xtrain,ytrain=X[:testduration],y[:testduration]
xtest,ytest=X[testduration:],y[testduration:]


Market_GoodRatio=sum(df['Good Buy Point?']==1)/len(df['Good Buy Point?'])#Good Buying Point Ratio in market is manully set to nearly 0.5 
ResultTable=ResultTable.append({'Stock':stock,'Method':'Market Good Buying Ratio','AvgScores':Market_GoodRatio,'StdScores':0},ignore_index=True)

#Compare and Plot the precision rate of each algorithm        
index=0
for method in method_list.loc[0,:]:
    clf = method
    #cv=TimeSeriesSplit(n_splits=3) #Time series test
    scores = cross_val_score(clf,xtrain, ytrain, cv=4,scoring='precision')
    print(scores[scores>0])
    series={'Stock':stock,'Method':method_list.columns[index],'AvgScores':scores[scores>0].mean(),'StdScores':scores[scores>0].std()}
    index=index+1
    ResultTable=ResultTable.append(series,ignore_index=True)

name_list= ['Market Good Buying Ratio']
name_list=np.append(name_list,method_list.columns)
num_list= ResultTable.loc[ResultTable['Stock']==stock]['AvgScores']
plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.title(stock+'\nPrecision Rate')
plt.show()
    
#Plot precission rate of each method 
index=0
for method in method_list.loc[0,:]:
     clf = method
     clf.fit(xtrain, ytrain)
     buypredicted = clf.predict_proba(xtest)
     precision, recall, threshold = precision_recall_curve(ytest, buypredicted[:,1])
     plot_precision_recall_vs_threshold(index,stock,method_list,precision, recall, threshold)
     plt.show()
     index=index+1
#%%       Naive Bayes       
clfbuy =GaussianNB(var_smoothing=1) 
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[testduration:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('Naive Bayes',dfplot,stock,0.93,0.99,0.015)
#%%  SVM             
clfbuy = svm.SVC(C=1,kernel='linear',probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[testduration:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('SVM Linear',dfplot,stock,0.93,0.99,0.015)
#%%  SVM             
clfbuy = svm.SVC(C=1,probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[testduration:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('SVM ',dfplot,stock,0.93,0.99,0.015)
#%%  SVM Poly         
clfbuy = svm.SVC(C=1,kernel='poly',probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[testduration:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('SVM Poly',dfplot,stock,0.93,0.99,0.015)

#%%  Random Forrest       
clfbuy =RandomForestClassifier(oob_score=True, random_state=30)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[testduration:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
plot_buy('Random Forrest',dfplot,stock,0.93,0.99,0.015)
