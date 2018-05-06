import numpy as np
import pandas as pd
import re
import string
import sys

def max_drawdown(s):
    drawdown = -sys.maxsize-1
    M = s[0]
    for i,v in enumerate(s):
        M = np.nanmax([M,v])
        drawdown = np.nanmax([M-v,drawdown])
    return drawdown

def max_drawup(s):
    return max_drawdown(-s)

def sharpe(s):
    mn = np.mean(s)
    mnsq = np.mean(s*s)
    return np.sqrt(250/len(s)) * mn / np.sqrt(mnsq-(mn**2))

def meanret(s):
    return round(np.mean(s), 4)

def uprat(s):
    try:
        return sum(s>0)/len(s)
    except ValueError:
        return 1.

def freq(s):
    ps = np.abs(np.fft.rfft(s))
    freq = np.fft.fftfreq(len(s), 1)
    ps/=sum(ps)
    return max(zip(freq[1:],ps[1:]), key=lambda x:x[1])[0]

def ampl(s):
    ps = np.abs(np.fft.rfft(s))
    freq = np.fft.fftfreq(len(s), 1)
    ps/=sum(ps)
    return max(zip(freq[1:],ps[1:]), key=lambda x:x[1])[1]

def max_Sharpe(s, min_days, up=True):
    d = np.diff(s)
    t1 = 0
    t2 = 0
    Shp = -sys.maxsize - 1
    coef = -1
    if (up):
        coef = 1
    for i in range(len(s)):
        for j in range(i+min_days,len(s)):
            Shp_test = Sharpe(d[i:j],i,j)
            if ((Shp_test * coef) > Shp):
                t1 = i
                t2 = j
                Shp = Shp_test
    return t1,t2,np.round((Shp*coef),4)
        
def date_intersect(df):
    dys = set(df[df.keys()[0]]['Norm_day'])
    for k in df:
        dys = dys.intersection(set(df[k]['Norm_day']))
    for k in df:
        df[k] = df[k][df[k]['Norm_day'].isin(dys)]
    
    return df

def matrix_vals(df,col_name='Settle12'):
    x = None
    for k in df:
        if (x is None):
            x = df[k][col_name].values
        else:
            x = np.row_stack((x,df[k][col_name].values))
    return x

                
                
    
    
