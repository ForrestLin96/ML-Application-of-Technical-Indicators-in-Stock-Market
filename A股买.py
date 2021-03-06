import pandas as pd
import datetime
import numpy as np
import tushare as ts
import talib as ta
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
from Afunction import buyjudge,plot_precision_recall_vs_threshold,plot_buy
from datasource import df,rawdata,stock,testduration,featurelist
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  
from sklearn.model_selection import GridSearchCV 
matplotlib.style.use('ggplot')

weights=1.1,1.1,0.5,0.5,1.2,1.1
softvoting= VotingClassifier(estimators=[('gbc',GaussianNB(var_smoothing=1)),
                                        ('sv_li',svm.SVC(kernel='linear', C=1,probability=True)),
                                        ('svm',svm.SVC(probability=True)),
                                        ('sv_po',svm.SVC(kernel='poly',degree=2,probability=True)),
                                        ('rnd',RandomForestClassifier(oob_score=True, random_state=30)),
                                        ('xgb',Xgb(reg_lambda=1))],voting='soft',weights=weights,n_jobs=-1)

method_name = [{
                # 'Random Forrest':RandomForestClassifier(),
                # 'Random Forrest30':RandomForestClassifier(oob_score=True, random_state=30),
                # 'Bayes(smo=1e-01)':GaussianNB(var_smoothing=1e-01),
                # 'Bayes(smo=0.5)':GaussianNB(var_smoothing=0.5),
                # 'Bayes(smo=1)':GaussianNB(var_smoothing=1),
                # 'Bayes(smo=2)':GaussianNB(var_smoothing=2),
                # 'SVC(C=1)':svm.SVC(probability=True),
                # 'SVC(linear, C=1)':svm.SVC(kernel='linear', C=1,probability=True),
                # 'SVC(poly, C=1)':svm.SVC(kernel='poly',degree=2,probability=True),
                # 'XGBT(λ=1)':Xgb(reg_lambda=1),#Result of parameter tunning in XGBPara.py
                # 'XGBT(λ=0.8)':Xgb(reg_lambda=0.8),
                # 'SoftVoting':softvoting,
                }]
method_list=pd.DataFrame(method_name)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])

df=buyjudge(df,duration=testduration)
 
df.dropna(axis=0, how='any', inplace=True)
xshow=df.iloc[testduration:,:].loc[:,featurelist]
xshow = preprocessing.MinMaxScaler().fit_transform(xshow)
if None in xshow[-1,:]:
    xshow=np.delete(xshow,-1,0)
X=df.loc[:,featurelist]
X = preprocessing.MinMaxScaler().fit_transform(X)
#X = preprocessing.StandardScaler().fit_transform(X)

y=df.loc[:,'Good buy Point?']
# Split train set and test set
xtrain,ytrain=X[:testduration],y[:testduration]
xtest,ytest=X[testduration:],y[testduration:]

Market_GoodRatio=sum(df['Good buy Point?'].iloc[:testduration,]==1)/len(df['Good buy Point?'].iloc[:testduration,])#Good buy Point Ratio in market is manully set to nearly 0.5 
ResultTable=ResultTable.append({'Stock':stock,'Method':'Market Good buy Ratio','AvgScores':Market_GoodRatio,'StdScores':0},ignore_index=True)

#Compare and Plot the precision rate of each algorithm        
index=0
for method in method_list.loc[0,:]:
    clf = method
    cv=TimeSeriesSplit(n_splits=4) #Time series test
    scores = cross_val_score(clf,xtrain, ytrain, cv=4,scoring='precision')
    print(scores[scores>0])
    series={'Stock':stock,'Method':method_list.columns[index],'AvgScores':scores[scores>0].mean(),'StdScores':scores[scores>0].std()}
    index=index+1
    ResultTable=ResultTable.append(series,ignore_index=True)

name_list= ['Market Good buy Ratio']
name_list=np.append(name_list,method_list.columns)
num_list= ResultTable.loc[ResultTable['Stock']==stock]['AvgScores']
plt.barh(range(len(num_list)), num_list,tick_label = name_list)
plt.title(stock+'\nPrecision Rate')
plt.show()
    
#Plot precision rate of each method 
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
buy_bayes =GaussianNB(var_smoothing=1) 
buy_bayes.fit(xtrain, ytrain)
buypredicted = buy_bayes.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodbuyProb']=buypredicted[:,1]
plot_buy('Naive Bayes',dfplot,stock,0.9,1,0.03)
#%%  SVM Linear       
model = svm.SVC(kernel='linear',probability=True)
grid = GridSearchCV(estimator=model, param_grid=[{'C': [1,5]}], cv=cv,scoring='precision')  
grid.fit(xtrain, ytrain)
print('网格搜索-度量记录：', grid.cv_results_)
buy_linear=grid.best_estimator_
buy_linear.fit(xtrain, ytrain)
buypredicted = buy_linear.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodbuyProb']=buypredicted[:,1]
plot_buy('SVM Linear',dfplot,stock,0.9,1,0.03)
#%%  SVM RBF          
model = svm.SVC(kernel='rbf',probability=True)
grid = GridSearchCV(estimator=model, param_grid=[{'C': [1,3,6],'gamma': [0.5, 0.1]}], cv=cv,scoring='precision')  
grid.fit(xtrain, ytrain)
print('网格搜索-度量记录：', grid.cv_results_)
buy_rbf=grid.best_estimator_
buy_rbf.fit(xtrain, ytrain)
buypredicted = buy_rbf.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodbuyProb']=buypredicted[:,1]
plot_buy('SVM RBF',dfplot,stock,0.9,1,0.03)
#%%  SVM Poly         
model = svm.SVC(kernel='poly',degree=2,probability=True)
grid = GridSearchCV(estimator=model, param_grid=[{'C': [1,3,6],'gamma': [0.5, 0.1]}], cv=cv,scoring='precision')  
grid.fit(xtrain, ytrain)
print('网格搜索-度量记录：', grid.cv_results_)
print('网格搜索-最佳度量值:', grid.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：', grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
buy_poly=grid.best_estimator_
buy_poly.fit(xtrain, ytrain)
buypredicted = buy_poly.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodbuyProb']=buypredicted[:,1]
plot_buy('SVM Poly',dfplot,stock,0.9,1,0.03)

#%%  Random Forrest       
buy_forrest =RandomForestClassifier(oob_score=True, random_state=30)
buy_forrest.fit(xtrain, ytrain)
buypredicted = buy_forrest.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodbuyProb']=buypredicted[:,1]
plot_buy('Random Forrest',dfplot,stock,0.9,1,0.03)
#%%  XGboost       
buy_xgb =Xgb(reg_lambda=1)
buy_xgb.fit(xtrain, ytrain)
buypredicted = buy_xgb.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodbuyProb']=buypredicted[:,1]
plot_buy('XGBoost',dfplot,stock,0.9,1,0.03)
#%% Soft Voting 
softvoting= VotingClassifier(estimators=[('gbc',GaussianNB(var_smoothing=1)),
                                        ('sv_li',buy_linear),
                                        ('svm_rbf',buy_rbf),
                                        ('sv_po',buy_poly),
                                        ('rnd',RandomForestClassifier(oob_score=True, random_state=30)),
                                        ('xgb',Xgb(reg_lambda=1))],voting='soft',weights=weights,n_jobs=-1)    
clfbuy =softvoting
clfbuy.fit(xtrain, ytrain)
buypredicted = clfbuy.predict_proba(xshow)    
dfplot=pd.DataFrame()
dfplot.loc[:,'close']=rawdata
dfplot.loc[:,'GoodbuyProb']=buypredicted[:,1]
plot_buy('Voting',dfplot,stock,0.9,1,0.03)
print(dfplot['GoodbuyProb'].max()-dfplot['GoodbuyProb'].min())
