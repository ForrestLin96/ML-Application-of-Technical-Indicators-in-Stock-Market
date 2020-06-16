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

def selljudge(df,cycle=10):
    #ticker_df['Max'] = ticker_df['Close'].rolling(window = cycle).max().shift(-cycle)
    df['Good Sell Point?'] =0
    df['10天最低价'] =df['Close'].rolling(window =cycle).min().shift(-cycle)/df['Close']
    df.loc[(df['10天最低价']<df['10天最低价'].quantile()),'Good Sell Point?'] = 1

def buyjudge(df,cycle=10):
    #ticker_df['Max'] = ticker_df['Close'].rolling(window = cycle).max().shift(-cycle)
    df['Good Buy Point?'] =0
    df['10天最高价'] =df['Close'].rolling(window =cycle).max().shift(-cycle)/df['Close']
    df.loc[(df['10天最高价']>df['10天最高价'].quantile()),'Good Buy Point?'] = 1
