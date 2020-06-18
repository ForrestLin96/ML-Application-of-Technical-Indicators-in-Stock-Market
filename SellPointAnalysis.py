import pandas as pd
import datetime
import numpy as np
from array import *
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
from sklearn import svm
from xgboost import XGBClassifier as Xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
matplotlib.style.use('ggplot')
from FunctionList import selljudge,stochastic_oscillator,calculate_k,calculate_dj,plot_precision_recall_vs_threshold,plot_sell
from sklearn import preprocessing   

stock='AAPL' #'AMZN','GOOG','FB','JNJ','V','PG','JPM','UNH','MA','INTC','VZ','HD','T','PFE','MRK','PEP']

method_name = [{
                # 'Random Forrest':RandomForestClassifier(),
                # 'Random Forrest30':RandomForestClassifier(oob_score=True, random_state=30),
                # 'SVC(C=1)':svm.SVC(probability=True),
                # 'SVC(C=1.5)':svm.SVC(C=1.5,probability=True),
                # 'SVC(linear, C=1)':svm.SVC(kernel='linear', C=1,probability=True),
                # 'SVC(poly, C=1)':svm.SVC(kernel='poly',probability=True),
                # 'XGBT(λ=0.8)':Xgb(reg_lambda=0.8),
                # 'XGBT(λ=1)':Xgb(reg_lambda=1),
                # 'Logistic':LogisticRegression()
                }]
method_list=pd.DataFrame(method_name)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])
start = datetime.datetime(2005,1,1)
end = datetime.datetime(2020,6,17)
testduration=-180
df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)
df_VIX = web.DataReader("^VIX", 'yahoo', start,end)


df=web.DataReader(stock, 'yahoo', start, end).drop(columns=['Adj Close'])
rawdata=df.iloc[testduration:]['Close']

#Selected indicators
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
df.loc[(df['K']<df['D']) & (df['K_prev']>df['D_prev']) & (df['D']>20),'Intersection'] = 1# Intersections: K go exceeds D   
df['# Inter 10-day'] = df['Intersection'].rolling(14).sum()# number of intersections during past 10 days    
df['Close/MA10']= df['Close']/df['Close'].rolling(10).mean()# df['MA10']= df['Close'].rolling(10) Moving Average of the past 10 days
df['Close/MA20']= df['Close']/df['Close'].rolling(20).mean()
df['Close/MA50']= df['Close']/df['Close'].rolling(50).mean()
df['Close/MA100']= df['Close']/df['Close'].rolling(100).mean()
df['Close/MA200']= df['Close']/df['Close'].rolling(200).mean()
df['VAR5']= df['Close_ROC'].rolling(5).std()
df['VAR10']= df['Close_ROC'].rolling(10).std()
selljudge(df,cycle=10)

#%%
featurelist=['# Inter 10-day','Intersection','MAVOL200','MAVOL20','MAVOL10','MAVOL5','SP500_ROC',
            'VIX_ROC','VIXMA5','VIXMA10','Close_ROC','rsv','K','D','J',
            'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','Close/MA10',
            'Close/MA20','Close/MA50','Close/MA100','Close/MA200','VAR5','VAR10']
xshow=df.iloc[testduration:,:].loc[:,featurelist]
xshow = preprocessing.MinMaxScaler().fit_transform(xshow)

df.dropna(axis=0, how='any', inplace=True)
#Retrive X and y 
X=df.loc[:,featurelist]
X = preprocessing.MinMaxScaler().fit_transform(X)

y=df.loc[:,'Good Sell Point?']
# Split train set and test set
xtrain,ytrain=X[:testduration],y[:testduration]
xtest,ytest=X[testduration:],y[testduration:]

Market_Sell_Ratio=sum(df['Good Sell Point?']==1)/len(df['Good Sell Point?'])#Good Selling Point Ratio in market is manully set to nearly 0.5 
ResultTable=ResultTable.append({'Stock':stock,'Method':'Market Good Selling Ratio','AvgScores':Market_Sell_Ratio,'StdScores':0},ignore_index=True)
#Compare and Plot the precision rate of each algorithm        
index=0
for method in method_list.loc[0,:]:
    clf = method
    #cv=TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(clf,xtrain, ytrain, cv=4,scoring='precision')
    print(scores[scores>0])
    series={'Stock':stock,'Method':method_list.columns[index],'AvgScores':scores[scores>0].mean(),'StdScores':scores[scores>0].std()}
    index=index+1
    ResultTable=ResultTable.append(series,ignore_index=True)

name_list= ['Market Good Selling Ratio']
name_list=np.append(name_list,method_list.columns)

num_list= ResultTable.loc[ResultTable['Stock']==stock]['AvgScores']
plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.title(stock+'\nPrecision Rate')
plt.show()
#Plot precission rate 
index=0
for method in method_list.loc[0,:]:
     clf = method
     clf.fit(xtrain, ytrain)
     sellpredicted = clf.predict_proba(xtest)
     precision, recall, threshold = precision_recall_curve(ytest, sellpredicted[:,1])
     plot_precision_recall_vs_threshold(index,'msft',method_list,precision, recall, threshold)
     plt.show()
     index=index+1

#%%  Visualize the points       
clfsell =Xgb(reg_lambda=0.8)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)   
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=rawdata
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
plot_sell('XGBoost',dfplot,stock,0.93,0.99,0.015)
#%%  Logistic      
clfsell =LogisticRegression()
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)   
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=rawdata
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
plot_sell('Logistic Regression',dfplot,stock,0.86,0.1,0.03)
#%%  SVM Poly       
clfsell =svm.SVC(kernel='poly',probability=True)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=rawdata
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
plot_sell('SVM Poly',dfplot,stock,0.93,0.99,0.015)
#%%  SVM linear     
clfsell =svm.SVC(probability=True)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=rawdata
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
plot_sell('SVM Linear',dfplot,stock,0.9,0.93,0.015)
#%%  Random Forrest       
clfsell =RandomForestClassifier()
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=rawdata
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
plot_sell('Random Forrest',dfplot,stock,0.93,0.99,0.015)
