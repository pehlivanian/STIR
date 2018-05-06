from pylab import *
import pandas as pd
from itertools import permutations, combinations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager
import string
import dateutil
import subprocess
import re
import os

def plot_GSCI(summ):

    fig = plt.figure()

    subplot_ind = 211
    
    fig.subplots_adjust(bottom=0.05)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.05)
    fig.subplots_adjust(left=.05)
    fig.subplots_adjust(right=.95)
    fig.subplots_adjust(right=.95)
    fig.subplots_adjust(wspace=0.1)
    axis = fig.add_subplot(subplot_ind)

    df = summ[['Date', 'PL']]
    df['PL'] = df['PL'].apply(pd.to_numeric)
    df = df.dropna(how='any')

    df.Date = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index(['Date'], inplace=True)

    df['PL'] = np.cumsum(df['PL'])
    
    df.plot(legend=True, grid=True, ax=axis)
    
    axis.set_title('Product{}: {}'.format(summ['Prod'].get_values()[0],
                                          summ['Strategy'].get_values()[0]))    
    axis.set_xlabel('Date')
    axis.set_ylabel('PL ($)')
    
    subplot_ind = 212
    
    fig.subplots_adjust(bottom=0.05)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.05)
    fig.subplots_adjust(left=.05)
    fig.subplots_adjust(right=.95)
    fig.subplots_adjust(right=.95)
    fig.subplots_adjust(wspace=0.1)
    axis = fig.add_subplot(subplot_ind)

    summ['PL'] = summ['PL'].apply(pd.to_numeric)
    summ.dropna(how='any')    
    summ['PL_EST'] = 0
    summ['PL_LIQ'] = 0
    est_mask = summ['SubStrategy'] == 'EST'
    liq_mask = summ['SubStrategy'] == 'LIQ'
    summ['PL_EST'][est_mask] = summ[est_mask]['PL']
    summ['PL_LIQ'][liq_mask] = summ[liq_mask]['PL']

    summ.Date = pd.to_datetime(summ['Date'], format='%Y-%m-%d')
    summ.set_index(['Date'], inplace=True)

    summ['PL_EST'] = np.cumsum(summ['PL_EST'])
    summ['PL_LIQ'] = np.cumsum(summ['PL_LIQ'])
    
    summ[['PL_EST', 'PL_LIQ']].plot(legend=True, grid=True, ax=axis)

    axis.set_title('Product{}: {}'.format(summ['Prod'].get_values()[0],
                                          summ['Strategy'].get_values()[0]))
    axis.set_xlabel('Date')
    axis.set_ylabel('PL ($)')

    return fig
    

def plot_spreads(df):

    first_row = next(iter(df.values()))[0:1]

    prod      = first_row['Prod1'].values[0]
    exch      = first_row['Exch1'].values[0]
    front_mth = first_row['Month1'].values[0]
    back_mth  = first_row['Month2'].values[0]
    offset    = first_row['Offset'].values[0]
                                                  
    TGTDIR = './'

    count = 0
    fileCount = 0

    subplotInd = 231

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.05)
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(bottom=0.05)
    fig.subplots_adjust(left=.05)
    fig.subplots_adjust(right=.95)
    fig.subplots_adjust(right=.95)
    fig.subplots_adjust(wspace=0.1)
    axis = fig.add_subplot(subplotInd)

    for key,val in df.items():
        dates = val.Date
        x = val.Settle12
        axis.plot(dates, x, linewidth=3)
    axis.set_title('Spread {}_{}{}-{}[{}]'.format(exch,prod,front_mth,back_mth,offset))
    axis.set_xlabel('Date')
    axis.set_ylabel('Level')

    subplotInd = 232

    dys = set(next(iter(df.values()))['Days_bef_FND'])
    for key,val in df.items():
        dys = dys.intersection( set(val['Days_bef_FND']))
    dys = sorted(list(dys))
    for key,val in df.items():
        df[key] = df[key][df[key]['Days_bef_FND'].isin(dys)]
    # x = np.zeros(len(dys))
    x = None
    for k,v in df.items():
        if x is None:
            x = df[k]['Settle12'].values
        else:
            x = row_stack((x,df[k]['Settle12'].values))

    axis = fig.add_subplot(subplotInd)
    level = x.mean(axis=0)
    std_level = sqrt(x.var(axis=0))
    axis.plot((dys), level, linewidth=3)
    axis.plot((dys), level + std_level, 'g-', linewidth=1)
    axis.plot((dys), level - std_level, 'g-', linewidth=1)
    for k in range(len(df)):
        axis.plot((dys), x[k,:], linewidth=.25)
    axis.set_title('Normalized Spread {}_{}-{}[{}]'.format(prod,front_mth,back_mth,offset))
    axis.set_xlabel('Days from LTD')
    axis.set_ylabel('Level')


    subplotInd = 233
    axis = fig.add_subplot(subplotInd)
    for k in range(len(df)):
        axis.plot(list(dys), x[k,:], linewidth=.25)
    axis.set_title('Normalized Spread {}_{}-{}'.format(prod,front_mth,back_mth))
    axis.set_xlabel('Days from LTD')
    axis.set_ylabel('Level')

    subplotInd = 234
    axis = fig.add_subplot(subplotInd)

    # level = plotUtils.bootstrap(x,all_but=2)
    
    m,n = x.shape
    ind = min(x.shape[1], 200)
    dys_trunc = list(dys)[(n-ind):]
    x_trunc = x[:,(n-ind):]
    level = x_trunc.mean(axis=0)
    std_level = sqrt(x_trunc.var(axis=0))
    axis.plot(dys_trunc, level, linewidth=3)
    # axis.plot(dys_trunc, level + std_level, 'g-', linewidth=1)
    # axis.plot(dys_trunc, level - std_level, 'g-', linewidth=1)
    # for k in range(len(df)):
    #     axis.plot(dys_trunc, x_trunc[k,:], linewidth=.25)
    axis.set_title('Normalized 200-day Spread {}_{}-{}[{}]'.format(prod,front_mth,back_mth,offset))
    axis.set_xlabel('Days from LTD')
    axis.set_ylabel('Level')

    subplotInd = 235
    axis = fig.add_subplot(subplotInd)

    min_days = {k:max(x.Norm_day) for k,x in df.items()}
    start_day = min([v for _,v in min_days.items()])
    end_day = 0
    diffs     = pd.DataFrame({k:np.diff(v[v['Norm_day'].isin(range(end_day,start_day))]['Settle12']) for k,v in df.items()})
    diff_summ = diffs.applymap(lambda x: x if x > 0 else x if x < 0 else x).apply(sum, axis=1)

    axis.plot(list(diff_summ.index), diff_summ.values, linewidth=2)
    axis.set_title('Runs Metric {}_{}-{}[{}]'.format(prod,front_mth,back_mth,offset))
    axis.set_xlabel('Days from LTD')
    axis.set_ylabel('Level')

    subplotInd = 236
    axis = fig.add_subplot(subplotInd)

    min_days = {k:max(x.Norm_day) for k,x in df.items()}
    start_day = min([v for _,v in min_days.items()])
    end_day = 0
    diffs     = pd.DataFrame({k:np.diff(v[v['Norm_day'].isin(range(end_day,start_day))]['Settle12']) for k,v in df.items()})
    diff_summ = diffs.applymap(lambda x: 1 if x > 0 else -1 if x < 0 else -x).apply(sum, axis=1)

    axis.plot(list(diff_summ.index), diff_summ.values, linewidth=2)
    axis.set_title('-1_1 Metric {}_{}-{}[{}]'.format(prod,front_mth,back_mth,offset))
    axis.set_xlabel('Days from LTD')
    axis.set_ylabel('Level')
        
    # fig.show()
    return fig
        
