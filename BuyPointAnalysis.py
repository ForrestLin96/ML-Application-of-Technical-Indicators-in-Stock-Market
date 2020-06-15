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
matplotlib.style.use('ggplot')

method_name = [{
                # 'Bayes(smo=1e-03)':GaussianNB(var_smoothing=1e-03),
                # 'Bayes(smo=1e-02)':GaussianNB(var_smoothing=1e-02),
                # 'Bayes(smo=1e-01)':GaussianNB(var_smoothing=1e-01),
                # 'Bayes(smo=0.5)':GaussianNB(var_smoothing=0.5),
                # 'Bayes(smo=1)':GaussianNB(var_smoothing=1),
                # 'Bayes(smo=2)':GaussianNB(var_smoothing=2),
                # 'SVC(C=1)':svm.SVC(probability=True),
                # 'SVC(C=1.5)':svm.SVC(C=1.5,probability=True),
                # 'SVC(linear, C=1)':svm.SVC(kernel='linear', C=1,probability=True),
                # 'SVC(poly, C=1)':svm.SVC(kernel='poly',probability=True),
                # 'XGBT(λ=1)':Xgb(reg_lambda=1),#Result of parameter tunning in XGBPara.py
                # 'XGBT(λ=1.2)':Xgb(reg_lambda=1.2),
                }]
method_list=pd.DataFrame(method_name)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])
start = datetime.datetime(2005,1,1)
end = datetime.date.today()
df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)
df_VIX = web.DataReader("^VIX", 'yahoo', start,end)

stocklist=['MSFT'] #Load ticker data'MSFT','AAPL','AMZN','GOOG','FB','JNJ','V','PG','JPM','UNH','MA','INTC','VZ','HD','T','PFE','MRK','PEP']
stock='MSFT'
#for stock in stocklist:
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
buyjudge(df,gain=0.024)
df.dropna(axis=0, how='any', inplace=True)#Get rid of rows with NA value

#Retrive X and y 
X=df.loc[:,['# Inter 10-day','Intersection','MAVOL200','MAVOL20','MAVOL10','MAVOL5','SP500_ROC',
            'VIX_ROC','VIXMA5','VIXMA10','Close_ROC','rsv','K','D','J',
            'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','Close/MA10',
            'Close/MA20','Close/MA50','Close/MA100','Close/MA200','VAR5','VAR10']]

X = preprocessing.MinMaxScaler().fit_transform(X) 
y=df.loc[:,'Good Buy Point?']

# Split train set and test set
xtrain,ytrain=X[:3500],y[:3500]
xtest,ytest=X[3500:],y[3500:]


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
for stock in stocklist:
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
     plot_precision_recall_vs_threshold(index,'msft',method_list,precision, recall, threshold)
     plt.show()
     index=index+1
#%%       Naive Bayes       
clfbuy =GaussianNB(var_smoothing=1) 
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3500:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
#for threshold in np.arange(0.65,0.683,0.003):
for threshold in np.arange(0.66,0.876,0.04):
    dfplot['Buy']=0
    dfplot['BuyPrice']=0
    dfplot.loc[(dfplot['GoodBuyProb']>threshold),'Buy'] = 1
    dfplot.loc[(dfplot['Buy']==1),'BuyPrice'] = dfplot['Close']
    buyratio=round(100*dfplot['Buy'].sum()/len(dfplot['Buy']),2)
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['BuyPrice']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Buy Point')
    plt.ylim([min(y1)-10, max(y1)+10])
    plt.title(stock+'\nNaive Bayes(smooth=1)\nThreshold='+str(round(threshold,3)))
    plt.figtext(0.35,0.2,'Buy Ratio='+str(buyratio)+'%' , fontsize=13)
    plt.figtext(0.65,0.8,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
    plt.legend(loc='upper left')
    plt.show()
#%%  SVM             
clfbuy = svm.SVC(C=1,kernel='linear',probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3500:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
for threshold in np.arange(0.62,0.8,0.035):
    dfplot['Buy']=0
    dfplot['BuyPrice']=0
    dfplot.loc[(dfplot['GoodBuyProb']>threshold),'Buy'] = 1
    dfplot.loc[(dfplot['Buy']==1),'BuyPrice'] = dfplot['Close']
    buyratio=round(100*dfplot['Buy'].sum()/len(dfplot['Buy']),2)
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['BuyPrice']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Buy Point')
    plt.ylim([min(y1)-10, max(y1)+10])
    plt.title(stock+'\nSVM Linear\nThreshold='+str(round(threshold,3)))
    plt.figtext(0.35,0.3,'Buy Ratio='+str(buyratio)+'%' , fontsize=13)
    plt.figtext(0.65,0.8,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
    plt.legend(loc='upper left')
    plt.show()
#%%  SVM             
clfbuy = svm.SVC(C=1,probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3500:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
for threshold in np.arange(0.63,0.67,0.01):
    dfplot['Buy']=0
    dfplot['BuyPrice']=0
    dfplot.loc[(dfplot['GoodBuyProb']>threshold),'Buy'] = 1
    dfplot.loc[(dfplot['Buy']==1),'BuyPrice'] = dfplot['Close']
    buyratio=round(100*dfplot['Buy'].sum()/len(dfplot['Buy']),2)
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['BuyPrice']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Buy Point')
    plt.ylim([min(y1)-10, max(y1)+10])
    plt.title(stock+'\nSVM \nThreshold='+str(round(threshold,3)))
    plt.figtext(0.35,0.3,'Buy Ratio='+str(buyratio)+'%' , fontsize=13)
    plt.figtext(0.65,0.8,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
    plt.legend(loc='upper left')
    plt.show()
#%%  Random Forrest            
clfbuy = svm.SVC(C=1,probability=True)
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xtest)    
dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3500:]['Close']
dfplot.loc[:,'GoodBuyProb']=buypredicted[:,1]
for threshold in np.arange(0.63,0.67,0.01):
    dfplot['Buy']=0
    dfplot['BuyPrice']=0
    dfplot.loc[(dfplot['GoodBuyProb']>threshold),'Buy'] = 1
    dfplot.loc[(dfplot['Buy']==1),'BuyPrice'] = dfplot['Close']
    buyratio=round(100*dfplot['Buy'].sum()/len(dfplot['Buy']),2)
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['BuyPrice']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Buy Point')
    plt.ylim([min(y1)-10, max(y1)+10])
    plt.title(stock+'\nSVM \nThreshold='+str(round(threshold,3)))
    plt.figtext(0.35,0.3,'Buy Ratio='+str(buyratio)+'%' , fontsize=13)
    plt.figtext(0.65,0.8,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
    plt.legend(loc='upper left')
    plt.show()