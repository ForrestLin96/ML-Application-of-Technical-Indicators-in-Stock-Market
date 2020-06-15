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
from FunctionList import selljudge,stochastic_oscillator,calculate_k,calculate_dj,plot_precision_recall_vs_threshold
from sklearn import preprocessing   

method_name = [{
                 'Random Forrest':RandomForestClassifier(),
                 'Random Forrest30':RandomForestClassifier(oob_score=True, random_state=30),
                # 'SVC(C=1)':svm.SVC(probability=True),
                # 'SVC(C=1.5)':svm.SVC(C=1.5,probability=True),
                # 'SVC(linear, C=1)':svm.SVC(kernel='linear', C=1,probability=True),
                # 'SVC(poly, C=1)':svm.SVC(kernel='poly',probability=True),
                # 'XGBT(λ=0.8)':Xgb(reg_lambda=0.8),
                # 'XGBT(λ=1)':Xgb(reg_lambda=1),
                }]
method_list=pd.DataFrame(method_name)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])
start = datetime.datetime(2005,1,1)
end = datetime.date.today()
df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)
df_VIX = web.DataReader("^VIX", 'yahoo', start,end)

stock='MSFT' #'AMZN','GOOG','FB','JNJ','V','PG','JPM','UNH','MA','INTC','VZ','HD','T','PFE','MRK','PEP']

df=web.DataReader(stock, 'yahoo', start, end).drop(columns=['Adj Close'])
rawdata=df
#Selected indicators
df['MAVOL200'] = df['Volume']/df['Volume'].rolling(200).mean()
df['MAVOL20'] = df['Volume']/df['Volume'].rolling(20).mean()
df['MAVOL10'] = df['Volume']/df['Volume'].rolling(10).mean()
df['MAVOL5'] = df['Volume']/df['Volume'].rolling(5).mean() 
df['SP500'] = df_SP500['Close']
df['SP500_ROC'] = 100*df['SP500'].diff(1)/df['SP500'].shift(1)
df['VIX'] = df_VIX['Close']
df['VIXMA3'] = df['VIX']/df['VIX'].rolling(5).mean()
df['VIXMA5'] = df['VIX']/df['VIX'].rolling(10).mean()
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
selljudge(df,loss=0.0149,cycle=10)
df.dropna(axis=0, how='any', inplace=True)#Get rid of rows with NA value
#Retrive X and y 
X=df.loc[:,['# Inter 10-day','Intersection','MAVOL200','MAVOL20','MAVOL10','MAVOL5','SP500_ROC','VIX_ROC','VIXMA3','VIXMA5','Close_ROC','rsv','K','D','J',
            'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','Close/MA10','Close/MA20','Close/MA50',
            'Close/MA100','Close/MA200','VAR5','VAR10']] 
ysell=df.loc[:,'Good Sell Point?']

X = preprocessing.MinMaxScaler().fit_transform(X) 
# split train and test data
xtrain,ytrain=X[:3500],ysell[:3500]
xtest,ytest=X[3500:],ysell[3500:]

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
#clfsell = svm.SVC(C=1,probability=True)
clfsell =Xgb(reg_lambda=1)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3500:]['Close']
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
for threshold in np.arange(0.87,0.92,0.01):
    dfplot['Sell']=0
    dfplot['SellPrice']=0
    dfplot.loc[(dfplot['GoodSellProb']>threshold),'Sell'] = 1
    dfplot.loc[(dfplot['Sell']==1),'SellPrice'] = dfplot['Close']
    Sellratio=round(100*dfplot['Sell'].sum()/len(dfplot['Sell']),2)
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['SellPrice']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Sell Point',color='blue')
    plt.ylim([min(y1)-10, max(y1)+10])
    plt.title(stock+'\nXGBoost(λ=0.8)\nThreshold='+str(round(threshold,3)))
    plt.figtext(0.35,0.2,'Sell Ratio='+str(Sellratio)+'%' , fontsize=13)
    plt.figtext(0.65,0.8,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
    plt.legend(loc='upper left')
    plt.show()
#%%  SVM Poly       
clfsell =svm.SVC(kernel='poly',probability=True)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3500:]['Close']
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
for threshold in np.arange(0.78,0.95,0.03):
    dfplot['Sell']=0
    dfplot['SellPrice']=0
    dfplot.loc[(dfplot['GoodSellProb']>threshold),'Sell'] = 1
    dfplot.loc[(dfplot['Sell']==1),'SellPrice'] = dfplot['Close']
    Sellratio=round(100*dfplot['Sell'].sum()/len(dfplot['Sell']),2)
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['SellPrice']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Sell Point',color='blue')
    plt.ylim([min(y1)-10, max(y1)+10])
    plt.title(stock+'\nSVM(Poly))\nThreshold='+str(round(threshold,3)))
    plt.figtext(0.35,0.3,'Sell Ratio='+str(Sellratio)+'%' , fontsize=13)
    plt.figtext(0.65,0.8,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
    plt.legend(loc='upper left')
    plt.show()
#%%  Random Forrest       
clfsell =RandomForestClassifier(oob_score=True, random_state=30)
clfsell.fit(xtrain, ytrain)
sellpredicted = clfsell.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3500:]['Close']
dfplot.loc[:,'GoodSellProb']=sellpredicted[:,1]
for threshold in np.arange(0.73,0.85,0.03):
    dfplot['Sell']=0
    dfplot['SellPrice']=0
    dfplot.loc[(dfplot['GoodSellProb']>threshold),'Sell'] = 1
    dfplot.loc[(dfplot['Sell']==1),'SellPrice'] = dfplot['Close']
    Sellratio=round(100*dfplot['Sell'].sum()/len(dfplot['Sell']),2)
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['SellPrice']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Sell Point',color='blue')
    plt.ylim([min(y1)-10, max(y1)+10])
    plt.title(stock+'\nRandomForrest)\nThreshold='+str(round(threshold,3)))
    plt.figtext(0.35,0.3,'Sell Ratio='+str(Sellratio)+'%' , fontsize=13)
    plt.figtext(0.65,0.8,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
    plt.legend(loc='upper left')
    plt.show()
