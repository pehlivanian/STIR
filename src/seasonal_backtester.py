from itertools import permutations, combinations
import abc
import numpy as np
import pandas as pd

import lib
import db
from data import (products, nrbys, months, years, fields, name_map, train_years, test_years, verify_years)
from lib.utils import SummStats, Singleton, Visitor
from lib import SeasonalParamList

import pdb

# In order:
# S = minimum Sharpe required
# R = minimum ratio up_days/down_days
# D = minimum number of days in window
# M = days before min(FND,LTD) for each series
# C = max year difference in spread contracts
# Y = minimum number of years required per study to train model

MIN_ROWS = 5

class Backtester(Visitor):
    def __init__(self, product, param_obj=SeasonalParamList[0]):
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

    def visit( self, element ):
        pass

    def _create_sql_table( self, tablename ):
        from sqlalchemy import Column, Integer, Float, String
        from sqlalchemy import MetaData
        from sqlalchemy import Table

        metadata = MetaData( self._DBConn._conn )
        # Create schema
        table = Table( tablename, metadata,
                    Column('ampl',        Float,       nullable=False),
                    Column('drawdn',      Float,       nullable=False),
                    Column('drawup',      Float,       nullable=False),
                    Column('freq',        Float,       nullable=False),
                    Column('meanret',     Float,       nullable=False),
                    Column('sharpe',      Float,       nullable=False),
                    Column('uprat',       Float,       nullable=False),
                    Column('strategy',    String(125), nullable=False),
                    Column('substrategy', String(125), nullable=False),
                    Column('pivot0',      Integer,     nullable=False),
                    Column('pivot1',      Integer,     nullable=False))

        metadata.create_all()    
    
    def strategy_summary_helper(self, price, front_mth, back_mth, offset, total_dols=1e6 ):

        counter = 0
        summary_all = pd.DataFrame()
        
        yrs = list(price.keys())
        num_yrs = len(yrs)

        num_days = {}
        for k,x in price.items():
            if isinstance( x, pd.DataFrame):
                num_days.update( {k:len(x.Settle12)} )
        num_days = sorted(num_days.items(), key=lambda x:x[1])

        # XXX
        # Possibility that min_days is 0
        min_days = {k:max(x.Norm_day) for k,x in price.items()}
        min_days = sorted(min_days.items(), key=lambda x:x[1])

        starting_day = min([x[1] for x in min_days])
        ending_day   = self._params.M + self._params.D

        [price.pop(x[0]) for x in min_days if x[1] < self._params.D]
        all_levels = pd.DataFrame()
        for year,levels in price.items():
            settle12 = levels[levels['Norm_day'].isin(range(ending_day,starting_day+1))][['Norm_day', 'Settle12']]
            settle12['Settle12'] = pd.to_numeric(settle12['Settle12'])
            settle12 = settle12.ffill()
            if all_levels.empty:
                all_levels = settle12
                all_levels['num_series'] = 1
            else:
                merged = pd.merge(all_levels, settle12, on=['Norm_day'], how='outer', suffixes=['_base', '_new'], indicator=True)
                merged['Settle12'] = merged.apply(lambda row: row.Settle12_base + row.Settle12_new if row._merge == 'both' else row.Settle12_base if row._merge == 'left_only' else row.Settle12_new, axis=1 )
                merged['num_series'] = merged.apply(lambda row: row.num_series + 1 if row._merge == 'both' else row.num_series, axis=1)
                merged = merged[['Norm_day', 'Settle12', 'num_series']]
                all_levels = merged

        all_levels['Settle12'] = all_levels['Settle12'] / all_levels['num_series']
        all_levels = all_levels[['Norm_day', 'Settle12']]

        pivots = lib.max_stats(all_levels['Settle12'])

        dols_per_tick = float(name_map['lotsize_map'][self._product]) * float(name_map['mult_map'][self._product])
        lots = total_dols / dols_per_tick
        all_levels['Position'] = 0
        all_levels['Position'][pivots[0]:pivots[1]] = lots
        all_levels['Position'][pivots[2]:pivots[3]] = -1 * lots

        all_levels['Price_mult'] = dols_per_tick
        all_levels = SummStats(all_levels).PL()

        strat_name = self._strategy_summary_strat_name( front_mth, back_mth, offset )
        
        est_levels   = pd.Series(all_levels['PL']).astype('float')
        est_levels   = est_levels[pivots[0]:pivots[1]]
        est_metrics  = pd.DataFrame({k:[v(est_levels)] for k,v in self._metrics_map.items()})
        est_metrics['substrategy'] = 'EST'
        est_metrics['strategy'] = strat_name
        est_metrics['pivot0'] = pivots[0]
        est_metrics['pivot1'] = pivots[1]
        
        liq_levels = pd.Series(all_levels['PL']).astype('float')
        liq_levels = liq_levels[pivots[2]:pivots[3]]
        liq_metrics = pd.DataFrame({k:[v(liq_levels)] for k,v in self._metrics_map.items()})
        liq_metrics['strategy'] = strat_name
        liq_metrics['substrategy'] = 'LIQ'
        liq_metrics['pivot0'] = pivots[2]
        liq_metrics['pivot1'] = pivots[3]

        summary = pd.concat( [ est_metrics, liq_metrics ] )

        table_name = pref = 'SEASONALS_'+self._product+'_'+self._exchange

        if table_name in self._DBConn.table_names():
            summary.to_sql(con=self._DBConn._conn, name=table_name, if_exists='append', index=False)
        else:
            self._create_sql_table( table_name )
            summary.to_sql(con=self._DBConn._conn, name=table_name, if_exists='append', index=False)
        print('Wrote to table: {!r}'.format(table_name))

    def _strategy_summary_strat_name( self, front_mth, back_mth, offset ):
        pref = 'SUMM_'+self._product+'_'+self._exchange+'_'
        return pref+front_mth+back_mth+str(offset)
        
    def _slice_summary_strat_name( self, front_mth, back_mth, offset, sim_day, start_day, end_day ):
        pref = 'SUMM_'+self._product+'_'+self._exchange+'_'
        suff = '['+str(start_day)+'-'+str(end_day)+']'
        return pref+front_mth+back_mth+str(offset)+'['+str(sim_day)+']'+suff
        
    def slice_summary_helper(self, price, front_mth, back_mth, offset ):

        counter = 0
        summary_all = pd.DataFrame()
        
        yrs = list(price.keys())
        num_yrs = len(yrs)

        num_days = {}
        for k,x in price.items():
            if isinstance( x, pd.DataFrame):
                num_days.update( {k:len(x.Settle12)} )
        num_days = sorted(num_days.items(), key=lambda x:x[1])

        # XXX
        # Possibility that min_days is 0
        min_days = {k:max(x.Norm_day) for k,x in price.items()}
        min_days = sorted(min_days.items(), key=lambda x:x[1])
        
        starting_day = min([x[1] for x in min_days])
        ending_day   = self._params.M + self._params.D
        
        [price.pop(x[0]) for x in min_days if x[1] < self._params.D]

        # Length of simulation in days
        for sim_day in range(self._params.M + self._params.D, starting_day - self._params.M - self._params.D):
        
            # Will generate the window [start_day, start_day - self._params.D + 1]
            for start_day in range(starting_day, sim_day - 1, -1):

                # Choose end_day so that range(start_day, end_day, -1)
                # contains the self._params.D days of the simulation window
                end_day = start_day - self._params.D

                # Form summary data frame
                strat_name = self._slice_summary_strat_name( front_mth, back_mth, offset, sim_day, start_day, end_day)

                levels = pd.DataFrame({k:v[v['Norm_day'].isin(range(end_day,start_day+1))]['Settle12'] for k,v in price.items()})
                levels = levels.apply(lambda row: pd.to_numeric(row), axis=1)
                pivots_all = pd.DataFrame({'turn_dates': levels.apply(lib.max_stats, axis=0) })

                diffs     = pd.DataFrame({k:np.diff(v[v['Norm_day'].isin(range(end_day,start_day+1))]['Settle12']) for k,v in price.items()})
                diffs     = diffs.apply(pd.to_numeric, axis=0)
                summary   = pd.DataFrame({k:diffs.apply(v, axis=0) for k,v in self._metrics_map.items()})

                # A little processing
                summary['strat'] = strat_name
                summary = summary[['strat'] + [col for col in summary.columns if not col == 'strat']]
                summary = summary.reset_index(level=0).rename(columns={'index' : 'year'})
                
                summary_all = pd.concat([summary_all, summary])

                print('Rows in summary_all: {}'.format(summary_all.shape[0]))

                if summary_all.shape[0] >= MIN_ROWS:
                    table_name = 'SEASONAL_SPREADS_' + self._product;
                    summary_all.to_sql(con=self._DBConn._conn, name=table_name, if_exists='append', index=False)
                    print('Wrote to table: {!r}'.format(table_name))
                    summary_all = pd.DataFrame()


    def create_summary(self, SLICE_SUMMARY=True):

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
                    if SLICE_SUMMARY:
                        self.slice_summary_helper(price, front_mth, back_mth, offset )
                    else:
                        self.strategy_summary_helper(price, front_mth,back_mth, offset )
                except Exception as e:
                    print(e)
                    continue

def seasonal_slice_backtest():
    for product in products:
        B = Backtester(product)
        B.create_summary(SLICE_SUMMARY=True)

def seasonal_strategy_backtest():
    # XXX
    products = ['C']
    for product in products:
        B = Backtester(product)
        B.create_summary(SLICE_SUMMARY=False)

