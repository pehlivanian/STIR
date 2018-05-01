from collections import namedtuple
from itertools import permutations, combinations
import abc
import numpy as np
import pandas as pd

import lib
import db
import data as config_data

products     = config_data.all_products()
nrbys        = config_data.all_nrbys()
months       = config_data.all_months()
years        = config_data.all_years()
fields       = config_data.all_fields()
name_map     = config_data.all_names()
train_years  = config_data.train_years()
test_years   = config_data.test_years()
verify_years = config_data.verify_years()


# In order:
# S = minimum Sharpe required
# R = minimum ratio up_days/down_days
# D = minimum number of days in window
# M = days before min(FND,LTD) for each series
# C = max year difference in spread contracts
# Y = minimum number of years required per study to train model

Param = namedtuple( 'Param', ['S', 'R', 'D', 'M', 'C', 'Y'])

MAX_ROWS = 10000

class Problem_0(object):
    def __init__(self):
        pass

    @staticmethod
    def params():
        return Param(S=3, R=.7, D=25, M=3, C=1, Y=10)

class Visitor(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def visit(self, element):
        pass

class Backtester(Visitor):
    def __init__(self, product, param_obj=Problem_0):
        self._product  = product
        self._exchange = name_map['exch_map'][product] 
        self._metrics_map = {'drawup'  : lib.max_drawup,
                             'drawdn'  : lib.max_drawdown,
                             'meanret' : lib.meanret,
                             'sharpe'  : lib.sharpe,
                             'uprat'   : lib.uprat,
                             'freq'    : lib.freq,
                             'ampl'    : lib.ampl}
        self._db_name = '_'.join(['STIR', self._exchange, self._product, 'SUMM'])
        self._SpreadObj = lib.Spread(product)
        self._DBConn = db.DBExt(self._product, db_name=self._db_name)
        self._params = param_obj.params()

    def report_helper(self, price, params, prod, exch, front_mth, back_mth, offset, metrics_map=None):

        counter = 0
        summary_all = pd.DataFrame()
        
        yrs = list(price.keys())
        num_yrs = len(yrs)
        
        num_days = {k:len(x.Settle12) for k,x in price.items()}
        num_days = sorted(num_days.items(), key=lambda x:x[1])
        
        min_days = {k:max(x.Norm_day) for k,x in price.items()}
        min_days = sorted(min_days.items(), key=lambda x:x[1])
        
        starting_day = min([x[1] for x in min_days])
        ending_day   = params.M + params.D
        
        [price.pop(x[0]) for x in min_days if x[1] < params.D]

        # Length of simulation in days
        for sim_day in range(params.M + params.D, starting_day - params.M - params.D):
        
            # Will generate the window [start_day, start_day - params.D + 1]
            for start_day in range(starting_day, sim_day - 1, -1):

                # Choose end_day so that range(start_day, end_day, -1)
                # contains the params.D days of the simulation window
                end_day = start_day - params.D

                for k,v in price.items():

                    # start_index, end_index for given year
                    start_index = v.index[v.Norm_day == start_day][0]
                    end_index   = v.index[v.Norm_day == end_day][0]
                
                    short_price = np.array(v[start_index:end_index]['Settle12'])

                # Form summary data frame
                strat_name = 'SUMM_'+prod+'_'+exch+'_'+front_mth+back_mth+str(offset)+'['+str(sim_day)+']'+'['+str(start_day)+'-'+str(end_day)+']'
                diffs     = pd.DataFrame({k:np.diff(v[v['Norm_day'].isin(range(end_day,start_day+1))]['Settle12']) for k,v in price.items()})      
                summary   = pd.DataFrame({k:diffs.apply(v, axis=0) for k,v in metrics_map.items()})

                # A little processing
                summary['strat'] = strat_name
                summary = summary[['strat'] + [col for col in summary.columns if not col == 'strat']]
                summary = summary.reset_index(level=0).rename(columns={'index' : 'year'})

                summary_all = pd.concat([summary_all, summary])

                print('Rows in summary_all: {}'.format(summary_all.shape[0]))
                
                if summary_all.shape[0] >= MAX_ROWS:
                    import pdb
                    pdb.set_trace()
                    table_name = 'spread_strat_summ_' + str(counter)
                    summary_all.to_sql(con=self._DBConn, name=table_name, if_exists='replace', index=False)
                    print('Wrote to table: {!r}'.format(table_name))
                    summary_all = pd.DataFrame()
                    counter += 1


    def create_summary(self):

        MAX_OFFSET = 2
        combs = list(combinations(months, 2))

        for (front_mth, back_mth) in permutations(months, 2):
            if (front_mth, back_mth) in combs:
                first_offset = 0
            else:
                first_offset = 1
                    
            for offset in range(first_offset, 1+MAX_OFFSET):
                try:
                    price = self._SpreadObj._create_study(front_mth, back_mth, offset)
                    self.report_helper(price, self._params, self._product, self._exchange, front_mth, back_mth, offset, self._metrics_map)
                except Exception as e:
                    continue
                        
            
if __name__ == '__main__':
    products = ['C']
    for product in products:
        B = Backtester(product)
        B.create_summary()

