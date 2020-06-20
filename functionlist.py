import pandas as pd
import datetime
import numpy as np
from array import *
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import precision_recall_curve
matplotlib.style.use('ggplot')

def plot_precision_recall_vs_threshold(index,stock,method_list,precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.xlim([0.5, 0.95])
    plt.ylim([0, 1])
    plt.title(stock+'\n'+method_list.columns[index])
#Define Stochastic Osciliator
def calculate_k(ticker_df,cycle, M1 ):
    Close = ticker_df['Close']
    highest_hi = ticker_df['High'].rolling(window = cycle).max()
    lowest_lo = ticker_df['Low'].rolling(window=10).min()
    ticker_df['rsv'] = (Close - lowest_lo)/(highest_hi - lowest_lo)*100
    ticker_df['K'] = ticker_df['rsv'].rolling(window=M1).mean()
    ticker_df['K']  =  ticker_df['K'] .fillna(50)
    ticker_df['K_diff'] = ticker_df['K'].diff()
    ticker_df['K_prev'] = ticker_df['K'] - ticker_df['K_diff']
    ticker_df['K_ROC'] = ticker_df['K']/ticker_df['K_prev']
    return ticker_df
def calculate_dj(ticker_df, M2 ):
    ticker_df['D'] = ticker_df['K'].rolling(window = M2).mean()
    ticker_df['D'] = ticker_df['D'].fillna(50)
    ticker_df['D_diff'] = ticker_df['D'].diff()
    ticker_df['D_prev'] = ticker_df['D'] - ticker_df['D_diff']
    ticker_df['D_ROC'] = ticker_df['D']/ticker_df['D_prev']
    ticker_df['J'] = M2*ticker_df['K']-(M2-1)*ticker_df['D']
    ticker_df['J_diff'] = ticker_df['J'].diff()
    ticker_df['J_prev'] = ticker_df['J'] - ticker_df['J_diff']
    ticker_df['J_ROC'] = ticker_df['J']/ticker_df['J_prev']
    return ticker_df
def stochastic_oscillator(ticker_df,cycle=12, M1=4, M2= 3):
    ticker_df = calculate_k(ticker_df,cycle,M1)
    ticker_df = calculate_dj(ticker_df, M2)
    return ticker_df
#Evenly separeate all days into good Selling points and poor Selling points

def selljudge(df,cycle=10,testduration=180):
    #ticker_df['Max'] = ticker_df['Close'].rolling(window = cycle).max().shift(-cycle)
    df['Good Sell Point?'] =0
    df['10天最低价'] =df['Close'].rolling(window =cycle).min().shift(-cycle)/df['Close']
    df.loc[(df['10天最低价']<df['10天最低价'][:-testduration,].quantile()),'Good Sell Point?'] = 1 #6/18新改动在train group里分好坏

def buyjudge(df,cycle=10,testduration=180):
    #ticker_df['Max'] = ticker_df['Close'].rolling(window = cycle).max().shift(-cycle)
    df['Good Buy Point?'] =0
    df['10天最高价'] =df['Close'].rolling(window =cycle).max().shift(-cycle)/df['Close']
    df.loc[(df['10天最高价']>df['10天最高价'].iloc[:-testduration,].quantile()),'Good Buy Point?'] = 1

def plot_buy(name,dfplot,stock,a=0.93,b=0.99,c=0.015):
    for ratio in np.arange(a,b,c):
        dfplot['Buy']=0
        dfplot['BuyPrice']=0
        dfplot.loc[(dfplot['GoodBuyProb']>dfplot['GoodBuyProb'].quantile(ratio)),'Buy'] = 1
        dfplot.loc[(dfplot['Buy']==1),'BuyPrice'] = dfplot['Close']
        buyratio=round(100*dfplot['Buy'].sum()/len(dfplot['Buy']),2)
        x=dfplot.index
        y1=dfplot['Close']
        y2=dfplot['BuyPrice']
        plt.plot(x, y1,'c',label='Price')
        plt.plot(x, y2, 'o', ms=4.5, label='Buy Point')
        plt.ylim([min(y1)*0.98, max(y1)*1.02])
        plt.title(stock+'\n'+name+'\nBuy Ratio='+str(buyratio)+'%, '+'Threshold='+str(round(dfplot['GoodBuyProb'].quantile(ratio),3)))
        #plt.figtext(0.35,0.3,'Buy Ratio='+str(buyratio)+'%' , fontsize=13)
        plt.figtext(0.7,0.5,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
        plt.figtext(0.7,0.45,'1 day:'+str(round(dfplot.iloc[-2,1],3)) , fontsize=13)
        plt.figtext(0.7,0.4,'2 day:'+str(round(dfplot.iloc[-3,1],3)) , fontsize=13)
        plt.legend(loc='upper left')
        plt.show()

def plot_sell(name,dfplot,stock,a=0.93,b=0.99,c=0.015):
    for ratio in np.arange(a,b,c):
        dfplot['Sell']=0
        dfplot['SellPrice']=0
        dfplot.loc[(dfplot['GoodSellProb']>dfplot['GoodSellProb'].quantile(ratio)),'Sell'] = 1
        dfplot.loc[(dfplot['Sell']==1),'SellPrice'] = dfplot['Close']
        sellratio=round(100*dfplot['Sell'].sum()/len(dfplot['Sell']),2)
        x=dfplot.index
        y1=dfplot['Close']
        y2=dfplot['SellPrice']
        plt.plot(x, y1,'c',label='Price')
        plt.plot(x, y2, 'o', ms=4.5, label='Sell Point',color='blue')
        plt.ylim([min(y1)*0.98, max(y1)*1.02])
        plt.title(stock+'\n'+name+'\nSell Ratio='+str(sellratio)+'%, '+'Threshold='+str(round(dfplot['GoodSellProb'].quantile(ratio),3)))
        #plt.figtext(0.35,0.3,'Sell Ratio='+str(Sellratio)+'%' , fontsize=13)
        plt.figtext(0.7,0.5,'Today:'+str(round(dfplot.iloc[-1,1],3)) , fontsize=13)
        plt.figtext(0.7,0.45,'1 day:'+str(round(dfplot.iloc[-2,1],3)), fontsize=13)
        plt.figtext(0.7,0.4,'2 day:'+str(round(dfplot.iloc[-3,1],3)) , fontsize=13)
        plt.legend(loc='upper left')
        plt.show()
