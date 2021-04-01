# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:41:33 2021

@author: mihir
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


stocks=['^NSEI','INDUSINDBK.NS','HDFCBANK.NS','ICICIBANK.NS','MARUTI.NS','TCS.NS','INFY.NS','ITC.NS','TATAMOTORS.NS','ASIANPAINT.NS']
end=dt.datetime.today()
start=dt.datetime.today()-dt.timedelta(3650)
stock_data={}

ohclv_data=pd.DataFrame()
for ticker in stocks:
    ohclv_data=yf.download(ticker,start,end)
    ohclv_data=ohclv_data.dropna()
    stock_data['{}'.format(ticker)]=ohclv_data


def RSI(DF,n):
    "function to calculate RSI"
    df = DF.copy()
    df['delta']=df['Close'] - df['Close'].shift(1)
    df['gain']=np.where(df['delta']>=0,df['delta'],0)
    df['loss']=np.where(df['delta']<0,abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n-1)*avg_gain[i-1] + gain[i])/n)
            avg_loss.append(((n-1)*avg_loss[i-1] + loss[i])/n)
    df['avg_gain']=np.array(avg_gain)
    df['avg_loss']=np.array(avg_loss)
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    df4=df.copy()
    return df4

def OBV(DF):
    df6=DF.copy()
    df6['change']=df6['Close'].pct_change()
    df6['direction']=np.where(df6['change']>0,1,-1)
    df6['direction'][0]=0
    df6['vol adj']=df6['Volume']*df6['direction']
    df6['obv']=df6['vol adj'].cumsum()
    return df6


def PSAR(DF):
    df9=DF.copy()
    opening_price=df9.iloc[:,0].tolist()
    low=df9.iloc[:,2].tolist()
    high=df9.iloc[:,1].tolist()
    AF=0.02
    psar=np.zeros(len(opening_price),dtype='float')
    count=0
    ep_up=0
    ep_down=0
    for i in range(0,len(opening_price)):
        
        if i==0:
            
            if opening_price[i+10]>opening_price[i]:
                e=[]
                f=[]
                for t in range(0,10):
                    e.append(low[t])
                    f.append(high[t])                     
                psar[10]=min(e)
                ep_up=max(f)
                psar[11]=psar[10]+abs((AF*(ep_up-psar[10])))
                if high[11]>ep_up:
                    count=count+1
                    AF=0.04
                    ep_up=high[11]
            else:
                e=[]
                f=[]
                for t in range(0,10):
                    e.append(high[t])
                    f.append(low[t])
                psar[10]=max(e)
                ep_down=min(f)
                psar[11]=(psar[10]-abs((AF*(ep_down-psar[10]))))
                if low[10]<ep_down:
                    ep_down=low[10]
                    count=count+1
                    AF=0.04
                       
        elif (i>=12 and psar[i-1]<=opening_price[i-1]):
            if (count<=9 and psar[i-2]>opening_price[i-2]) :
                count=0
                AF=0.02
                ep_up=high[i]
            
            psar[i]=psar[i-1]+abs((AF*(ep_up-psar[i-1])))
                
            if(high[i]>ep_up and count<9):
                ep_up=high[i]
                count=count+1
                AF=AF+ (0.02*count)
                
            if(high[i]>ep_up):
                ep_up=high[i]
            
            if count==9:
                AF=0.2
    
                    
        elif (i>=12 and psar[i-1]>opening_price[i-1]):
            if (count<=9 and psar[i-2]<opening_price[i-2] ):
                count=0
                AF=0.02
                ep_down=low[i]
                
            psar[i]=psar[i-1]-abs((AF*(ep_down-psar[i-1])))
                
            if(low[i]<ep_down and count<9):
                ep_down=low[i]
                count=count+1
                AF=AF+ (0.02*count)
                
            if (low[i]<ep_down):

                ep_down=low[i]
            
            if count==9:
                AF=0.2
            
    psar=np.asarray(psar).reshape(len(psar),1)
    df9['PSAR']=psar
    return df9


signal_data={}
for ticker in stocks:
    df1=PSAR(stock_data[ticker])
    df2=RSI(stock_data[ticker], 14)
    df3=OBV(stock_data[ticker])
    x=pd.DataFrame()
    y=pd.DataFrame()
    x=stock_data[ticker].copy()
    p=np.asarray(df1.iloc[-len(x):,6].values)
    p=p.reshape(len(p),1)
    x['PSAR']=df1.iloc[-len(p):,6].values
    a=np.asarray(df2.iloc[-len(x):,12].values)
    a=a.reshape(len(a),1)
    x['RSI']=df2.iloc[-len(a):,12].values
    b=np.asarray(df3.iloc[-len(x):,9].values)
    b=b.reshape(len(b),1)
    x['OBV']=df3.iloc[-len(b):,9].values
    x=x.dropna()
    p=x['PSAR'].tolist()
    o=x['Open'].tolist()
    psar_trend=[]
    for i in range(0,len(p)):
        if p[i] < o[i]:
            psar_trend.append(1)
        else:
            psar_trend.append(0)
    psar_trend=pd.DataFrame(psar_trend)
    x['PSAR_trend']=psar_trend.iloc[-len(p):,0].values
    rsi=x['RSI'].tolist()
    rsi_trend=[]
    for i in range(0,len(rsi)):
        if rsi[i]>=60:
            rsi_trend.append(0)
        else:
            rsi_trend.append(1)
    rsi_trend=pd.DataFrame(rsi_trend)
    x['RSI_trend']=rsi_trend.iloc[-len(rsi):,0].values
    obv=x['OBV'].tolist()
    obv_value=[]
    obv_trend=[]
    obv_trend.append(0)
    obv_value.append(0)
    for i in range(0,len(obv)-1):
        z=obv[i+1]-obv[i]
        obv_value.append(z)
    for i in range(0,len(obv_value)-1):
        if obv_value[i+1]>obv_value[i]:
            obv_trend.append(1)
        else:
            obv_trend.append(0)
    obv_trend=pd.DataFrame(obv_trend)
    x['OBV_trend']=obv_trend.iloc[-len(obv):,0].values
    
    signal=[]
    rsi_trend=rsi_trend[0].tolist()
    psar_trend=psar_trend[0].tolist()
    obv_trend=obv_trend[0].tolist()
    for i in range(0,len(psar_trend)):
        if (psar_trend[i]==1 and rsi_trend[i]==1) or (psar_trend[i]==1 and obv_trend[i]==1):
            signal.append(1)
        elif (psar_trend[i]==0 and rsi_trend[i]==0) or (psar_trend[i]==0 and obv_trend[i]==0):
            signal.append(-1)
        else:
            signal.append(0)
    signal=pd.DataFrame(signal)
    x['SIGNAL']=signal.iloc[-len(signal):,0].values
    signal_data['{}'.format(ticker)]=x


df1.iloc[-100:,[0,6]].plot()
df2.iloc[-100:,12].plot()
