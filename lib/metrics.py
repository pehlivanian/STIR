import numpy as np
import pandas as pd
import re
import string
import sys

def max_stats(arr):
    M = np.finfo(np.float32).min
    m = np.finfo(np.float32).max
    drawup = np.finfo(np.float32).min
    drawdn = np.finfo(np.float32).min

    for ind,d in enumerate(arr):
        if M-d > drawdn:
            drawdn = M-d
            iM,jM = max_ind,ind
        if d-m > drawup:
            drawup = d-m
            im,jm = min_ind,ind
        if d > M:
            max_ind = ind
            M = d
        if d < m:
            min_ind = ind
            m = d
            
    # [im. jm] ~ drawup
    # [iM, jM] ~ drawdn
    return im,jm,iM,jM,drawup,drawdn

def max_drawdown(s, mult=1):
    drawdown = -sys.maxsize-1
    M = s.get_values()[0]
    for i,v in enumerate(s):
        M = np.nanmax([M,v])
        drawdown = np.nanmax([M-v,drawdown])
    return mult*drawdown

def max_drawdown_levels(s, mult=1):
    cs = np.cumsum(s)
    drawdown = -sys.maxsize-1
    M = cs.get_values()[0]
    for i,v in enumerate(cs):
        M = np.nanmax([M,v])
        drawdown = np.nanmax([M-v,drawdown])
    return mult*drawdown

def max_drawup(s, mult=1):
    return mult*max_drawdown(-s)

def max_drawup_levels(s, mult=1):
    return mult*max_drawdown_levels(-s)

def sharpe(s):
    return np.sqrt(252)*np.nanmean(s)/np.nanstd(s)

def sharpe_levels(s):
    return np.sqrt(252)*np.nanmean(s)/np.nanstd(s)

def meanret(s, mult=1):
    return mult*round(np.nanmean(s), 4)

def meanret_levels(s, mult=1):
    return mult*round(np.nanmean(s), 4)

def uprat(s):
    try:
        return sum(s>0)/len(s)
    except ValueError:
        return 0.

def uprat_levels(s):
    try:
        return sum(s>0)/len(s)
    except ValueError:
        return 0.

def nonnegrat(s):
    try:
        return sum(s>=0)/len(s)
    except ValueError:
        return 0.

def nonnegrat_levels(s):
    try:
        return sum(s>=0)/len(s)
    except ValueError:
        return 0.
    
def freq(s):
    ps = np.abs(np.fft.rfft(s.fillna(0)))
    freq = np.fft.fftfreq(len(s), 1)
    ps/=sum(ps)
    try:
        m = max(zip(freq[1:],ps[1:]), key=lambda x:x[1])[0]
    except ValueError:
        m = np.nan
    return m

def freq_levels(s):
    ps = np.abs(np.fft.rfft(s.fillna(0)))
    freq = np.fft.fftfreq(len(s), 1)
    ps/=sum(ps)
    try:
        m = max(zip(freq[1:],ps[1:]), key=lambda x:x[1])[0]
    except ValueError:
        m = np.nan
    return m

def ampl(s):
    ps = np.abs(np.fft.rfft(s))
    freq = np.fft.fftfreq(len(s), 1)
    ps/=sum(ps)
    try:
        m = max(zip(freq[1:],ps[1:]), key=lambda x:x[1])[1]
    except ValueError:
        m = np.nan
    return m

def ampl_levels(s):
    ps = np.abs(np.fft.rfft(s))
    freq = np.fft.fftfreq(len(s), 1)
    ps/=sum(ps)
    try:
        m = max(zip(freq[1:],ps[1:]), key=lambda x:x[1])[1]
    except Exception:
        m = np.nan
    return m

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

                
                
    
    
