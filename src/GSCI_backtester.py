from collections import namedtuple
import abc
import datetime
import bisect
import logging
import pandas as pd
import numpy as np
from functools import partial
from pymongo import MongoClient
from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy_utils import database_exists, create_database
from dateutil.relativedelta import relativedelta
from data import (products, nrbys, months, years, fields, name_map, train_years, test_years, verify_years)
from lib.utils import ( contract_month_map, Credentials, SummStats, Singleton, Visitor,
                        US_Federal_Calendar, bmth_us, bday_us, bus_day_add)
from lib import GSCI_params
from data import GSCIData
import lib
import db

# Parameters, in order
# DB = num closes before first roll close that we start accumulating position
# acc_method = position accumulation method:
#        { 'linear' : proportional accumulation
#          'bullet' : immediate accumulation }
# DA = num closes after last roll close that we start liquidating position
# liq_method = position liquidation method:
#        { 'linear' : proportional accumulation
#          'bullet' : immediate accumulation }
# DBE = num days before expiry we must be flat

GSCI_data     = GSCIData()
GSCI_comp     = GSCI_data.GSCI_comp
GSCI_products = GSCI_data.products

ParamList     = GSCI_params.GSCIParamList
POS_COLS = ['Date', 'Prod1', 'Month1', 'Month2', 'Offset','Settle1', 'Settle2', 'Position', 'Settle12', '_merge']

def roll_spread_months(product, month):
    ''' Returns front month, back month as contract letter
    for the product roll during the give calendar month
    letter.
    
    offset      == 1 ===> front month occurs in current calendar year, back
                          month appears in following year
    year_offset == 1 ===> front month and back month occur in following calendar
                          year
    '''
    
    front_month = str(month)
    back_month  = str(1+(month%12))
    GSCI_comp_row = GSCI_comp.loc[GSCI_comp['Product'] == product]

    offset, year_offset = 0,0

    months = ( front_month, back_month)
    front_month_str, back_month_str = [ GSCI_comp_row[month].iloc[0] for month in months ]
    
    offset = 1 if contract_month_map[front_month_str] > contract_month_map[back_month_str] else 0
    year_offset = 1 if contract_month_map[front_month_str] < month else 0

    return front_month_str, back_month_str, offset, year_offset

def roll_dates_by_month(yr, mth):
    ''' roll dates by month, with rule specifiation
    '''
    
    yr, mth = int(yr), int(mth)
    dts = [dtime.date() for dtime in pd.date_range(datetime.date(yr, mth, 1) +
                                                   relativedelta(days=-2) + bmth_us, periods=9, freq=bday_us)]

    return dts[4], dts[-1]

def create_roll_dates(days_offset, roll_date, establish=True):
    ''' Create roll dates, with rule specification
    '''
    _dates = list()
    
    if establish:
        last_date = roll_date
        
        while days_offset > 0:
            last_date -= bday_us
            days_offset -= 1
            _dates.append(last_date)
        _dates.reverse()
        df_dates = pd.DataFrame({'Date' : [dt.date() for dt in _dates] })
    else:
        first_date = roll_date

        while days_offset > 0:
            days_offset -= 1
            _dates.append(first_date)
            first_date += bday_us

        df_dates = pd.DataFrame( {'Date' : [dt.date() if not type(dt).__name__ == 'date' else
                                            dt for dt in _dates] })

    return df_dates

def create_pos( dates, price, lots, foll_year, product_dols, dols_per_tick, establishing):
    ''' Create position dataframe
    '''

    if establishing:
        pos_df = pd.DataFrame({ 'Position' : np.linspace(0,      -1*lots, 1+dates.shape[0])[1:] })
    else:
        pos_df = pd.DataFrame({ 'Position' : np.linspace(lots,   0,       1+dates.shape[0])[:-1]})
    pos_df = pd.concat([dates, pos_df], axis=1)
    
    price_pos = pd.merge(pos_df, price[foll_year], how='left', indicator=True)[POS_COLS]

    price_pos = price_pos.dropna(how='any')

    # Recalculate weights for new size, if necessary
    if price_pos.shape[0] != dates.shape[0]:
        if establishing:
            price_pos['Position'] = np.linspace(0,    -1*lots, 1+price_pos.shape[0])[1:]
        else:
            price_pos['Position'] = np.linspace(lots, 0,       1+price_pos.shape[0])[:-1]
        
    price_pos['Dols']          = product_dols
    price_pos['Price_mult']    = dols_per_tick
    price_pos['SubStrategy']   = 'EST' if establishing else 'LIQ'
    price_pos.rename(columns   = { 'Prod1' : 'Prod' }, inplace=True)

    return price_pos

class Backtester(Visitor):
    def __init__(self, product, param_obj=GSCI_params.GSCIParamList[0]):
        self._product     = product
        self._exchange    = name_map['exch_map'][self._product]
        self._credentials = Credentials()
        self._metrics_map = {'drawupK'   : partial(lib.max_drawup_levels,   mult=1e-3),
                             'drawdnK'   : partial(lib.max_drawdown_levels, mult=1e-3),
                             'meanretK'  : partial(lib.meanret_levels,      mult=1e-3),
                             'sharpe'    : lib.sharpe_levels,
                             'updnrat'   : lib.uprat_levels,
                             'nonnegrat' : lib.nonnegrat_levels,
                             'freq'      : lib.freq_levels,
                             'ampl'      : lib.ampl_levels}
        self._SpreadObj   = lib.Spread(product)        
        self._db_name     = '_'.join(['STIR', self._exchange, self._product, 'SUMM'])
        self._paramObj    = param_obj
        self._params      = param_obj.params
        self._props       = param_obj.props
        self._props_name  = param_obj.props_name

        credentials       = self._credentials.DB_creds(db_override=self._db_name)
        self._DBConn      = self._DB_conn(*credentials)
        
    @staticmethod
    def _DB_conn(user, passwd, db):
        conn = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user,passwd,db))
        if not database_exists(conn.url):
            create_database(conn.url)
        return conn

    def __repr__(self):
        r = '_'.join([ self._product,
                       self._exchange,
                       str( self._params.DB ),
                       self._params.acc_method,
                       str( self._params.DA ),
                       self._params.liq_method ] )
        if self._props:
            r += '_{}'.format(self._props_name)
        return r

    def visit(self, element):
        pass

    def backtest_helper(self, total_dollars=1e6):

        if self._product not in GSCI_products:
            return None, None, None, None

        portfolio_results = pd.DataFrame()

        RPDW = GSCI_comp.loc[GSCI_comp['Product'] == self._product]['2017_RPDW'].iloc[0]
        days_before, est_method, days_after, liq_method, days_before_expiration = self._params

        S = lib.Spread(self._product)

        for month_num in range(1,13):
            front_month, back_month, offset, year_offset = roll_spread_months(self._product, month_num)

            if front_month == back_month and offset == 0:
                log_msg = 'Front month == back month for month number: {}'.format(month_num)
                logging.warn(log_msg)
                continue

            try:
                price = S._create_study(front_month, back_month, offset, train_only=False)
            except OperationError:
                logging.warn('Sql connection down for (month, year): ({}, {})'.format( month_num, year))

            all_years = sorted(price.keys()) 

            for year in all_years:
                curr_year,foll_year = year, str(int(year)+year_offset)
                if int(foll_year) > max([int(yr) for yr in all_years]):
                    log_msg = 'Spread window for (year, month number): ({}, {}) is beyond test period; skipping'.format(curr_year, month_num)
                    logging.warn(log_msg)
                    continue

                if not isinstance(price[year], pd.DataFrame):
                    log_msg = 'Price data for year {} missing'.format(year)
                    logging.warn(log_msg)
                    continue
                
                start_roll_date, end_roll_date = roll_dates_by_month(curr_year, month_num)

                # We will use an approximation to the weight of each component in the GSCI
                # The documents state that
                # IDW_d = Sum_c{CPW_D^d * DCRP_d^c)
                #
                # where
                # c = the designated contract
                # d = the S&P GSCI business day on which the calculation is made
                # DCRP = the Daily Contract Reference Price
                #
                # We will use the settlement price of the front month on the first day of
                # the establishing leg of the strategy in the above formula to calculate
                # holdings according to the percentage given in the RPDW column.

                est_dates_df  = create_roll_dates(days_before, start_roll_date, establish=True)
                liq_dates_df  = create_roll_dates(days_after, end_roll_date, establish=False)
                                
                # Establishing position: calculation of number of units
                try:
                    DCRP_ref_date = est_dates_df['Date'].get_values()[0]
                    DCRP = float(price[foll_year][price[foll_year]['Date'] == DCRP_ref_date]['Open1'].get_values()[0])
                except KeyError:
                    logging.warn(
                        'year {} not contained in price matrix - needed for (month,year): ([], {})calculation'.format( foll_year,month_num, year))
                    continue
                except Exception as e:
                    days_back = 1
                    while days_back < 10:
                        try:
                            DCRP_ref_date = bus_day_add(est_dates_df['Date'].get_values()[0], -days_back)
                            DCRP = float(price[foll_year][price[foll_year]['Date'] == DCRP_ref_date]['Open1'].get_values()[0])
                            days_back += 1
                            break
                        except IndexError as e:
                            days_back += 1

                # Round to nearest integer
                # lots              = round(total_dollars * (RPDW / 100) / float(name_map['lotsize_map'][self._product])
                #                           / DCRP / float(name_map['mult_map'][self._product]), 0)

                # Don't
                lots              = total_dollars * (RPDW / 100) / float(name_map['lotsize_map'][self._product]) / \
                                    DCRP / float(name_map['mult_map'][self._product])
                
                product_dols      = total_dollars * (RPDW / 100)
                dols_per_tick     = float(name_map['lotsize_map'][self._product]) * float(name_map['mult_map'][self._product])

                # Establishing position
                est_price_pos = create_pos(est_dates_df, price, lots, foll_year, product_dols, dols_per_tick, True)
                est_price_pos = SummStats(est_price_pos).PL()

                # Liquidating position
                if self._props.get('liq_if_est_money', False):
                    est_levels   = pd.Series(est_price_pos['PL']).astype('float')
                    est_metrics  = pd.DataFrame({k:[v(est_levels)] for k,v in self._metrics_map.items()})
                    # No EST positions for this strategy
                    est_price_pos = None                    
                    liq_price_pos = None
                    
                    if est_metrics.loc[0].sharpe > self._props.get('liq_cutoff'):
                        liq_price_pos = create_pos(liq_dates_df, price, lots, foll_year, product_dols, dols_per_tick, False)
                        liq_price_pos = SummStats(liq_price_pos).PL()
                else:
                    liq_price_pos = create_pos(liq_dates_df, price, lots, foll_year, product_dols, dols_per_tick, False)
                    liq_price_pos = SummStats(liq_price_pos).PL()                    

                portfolio_results = pd.concat([portfolio_results, est_price_pos, liq_price_pos])

        portfolio_results['RPDW'] = RPDW
        portfolio_results['Strategy'] = self.__repr__()

        portfolio_results = portfolio_results.sort_values('Date')
        portfolio_results = portfolio_results.dropna(how='any', subset=['Date'])

        levels  = pd.Series(portfolio_results['PL']).astype('float')

        if not levels.empty:

            summ_table_name = '_'.join(['GSCI_strat_summ', self.__repr__()])
            portfolio_results.to_sql(con=self._DBConn, name=summ_table_name, if_exists='replace', index=False)
            logging.info('Wrote to table {}'.format( summ_table_name ))

            metrics  = pd.DataFrame({k:[v(levels)] for k,v in self._metrics_map.items()})
            metrics_table_name = '_'.join(['GSCI_strat_metrics', self.__repr__()])
            metrics.to_sql(con=self._DBConn, name=metrics_table_name, if_exists='replace', index=False)
            logging.info('Wrote to table {}'.format( metrics_table_name ))

            return portfolio_results, metrics, summ_table_name, metrics_table_name

        return None, None, None, None
    
def GSCI_backtest(products=products):

    for product in products:
        for ind,param_obj in enumerate(ParamList):
            print('Product: {} ParamObj: {}'.format( product, ind ))
            B = Backtester(product, param_obj=param_obj)
            B.backtest_helper()
            
