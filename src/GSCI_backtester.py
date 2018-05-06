from collections import namedtuple
import abc
import datetime
import bisect
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import CustomBusinessMonthBegin
from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy_utils import database_exists, create_database
from dateutil.relativedelta import relativedelta

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


# In order
# DB = num closes before first roll close that we start accumulating position
# acc_method = position accumulation method:
#        { 'linear' : proportional accumulation
#         'bullet' : immediate accumulation }
# DA = num closes after last roll close that we start liquidating position
# liq_method = position liquidation method:
#        { 'linear' : proportional accumulation
#         'bullet' : immediate accumulation }

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Credentials(metaclass=Singleton):

    def __init__(self):
        client = MongoClient()

        self._coll = client.STIR.get_collection('credentials')
    
    def DB_creds(self, db_override=None):
        creds = self._coll.find_one()
        if db_override:
            return creds['TS_DB']['user'], creds['TS_DB']['passwd'], db_override
        else:
            return creds['TS_DB']['user'], creds['TS_DB']['passwd'], creds['TS_DB']['db']

    def quandl_creds(self):
        creds = self._coll.find_one()
        return creds['quandl']['api_key']

Param = namedtuple('Param', ['DB', 'acc_method', 'DA', 'liq_method'])

contract_month_map = dict(zip('FGHJKMNQUVXZ', range(1,13)))

GSCI_comp = { 'Product' : ['W', 'KW', 'C', 'S', 'KC', 'SB', 'CC', 'CT', 'LH', 'LC', \
                           'FC', 'CL', 'HO', 'RB', 'B', 'G', 'NG', 'MAL', 'MCU', \
                           'MNI', 'MPB', 'MZN', 'GC', 'SI'],
              'Exchage' : ['CME'] + ['KBT'] + (['CME'] * 2) + (['ICE'] * 4) + \
                          (['CME'] * 6) + (['ICE'] * 2) + ['CME'] + (['LME'] * 5) \
                          + (['CME'] * 2),
              '2016_PDW' : [3.53, 0.88, 4.23, 2.95, 0.94, 1.59, 0.45, 1.19, 2.30, \
                            4.79, 1.55, 23.04, 5.21, 5.31, 20.43, 5.82, 3.24, 2.88,\
                            3.85, 0.60, 0.60, 0.88, 3.24, 0.41],
              '2017_RPDW' : [3.90, 1.09, 5.49, 3.78, 1.03, 2.47, 0.58, 1.54, 2.66, \
                             5.08, 1.49, 22.80, 4.06, 4.67, 16.49, 4.91, 3.32, 3.25,\
                             4.06, 0.66, 0.74, 1.00, 4.39, 0.56],
              '1' :  (['H'] * 8) + (['G'] * 2) + ['H'] + (['G'] * 3) + \
                                          ['H'] + (['G'] * 8) + ['H'],
              '2' :  (['H'] * 8) + (['J'] * 2) + (['H'] * 4) + ['J'] + \
                                          (['H'] * 7) + ['J'] + ['H'],
              '3' :  (['K'] * 8) + (['J'] * 6) + ['K'] + (['J'] * 8) + \
                                          ['K'],
              '4' :  (['K'] * 8) + (['M'] * 2) + (['K'] * 4) + ['M'] + \
                                          (['K'] * 7) + ['M'] + ['K'],
              '5' :  (['N'] * 8) + (['M'] * 2) + ['Q'] + (['M'] * 3) + \
                                          ['N'] + (['M'] * 8) + ['N'],
              '6' :  (['N'] * 9) + (['Q'] * 2) + (['N'] * 3) + ['Q'] + \
                                          (['N'] * 7) + ['Q'] + ['N'],
              '7' :  (['U'] * 3) + ['X'] + ['U']+ ['V'] + ['U'] + ['Z'] + \
                                          (['Q'] * 6) + ['U'] + (['Q'] * 8) + ['U'],
              '8' :  (['U'] * 3) + ['X'] + ['U']+ ['V'] + ['U'] + ['Z'] + \
                                          (['V'] * 2) + (['U'] * 4) + ['V'] + (['U'] * 7) + \
                                           ['Z'] + ['U'],
              '9' :  (['Z'] * 3) + ['X'] + ['Z']+ ['V'] + (['Z'] * 2) + \
                                          (['V'] * 7) + (['V'] * 7) + (['Z'] * 2),
              '10' : (['Z'] * 3) + ['X'] + ['Z']+ ['H'] + (['Z'] * 4) + \
                                          (['X'] * 4) + ['Z'] + (['X'] * 7) + (['Z'] * 2),
              '11' : (['Z'] * 3) + ['F'] + ['Z']+ ['H'] + (['Z'] * 4) + \
                                          ['F'] + (['Z'] * 3) + ['F'] + (['Z'] * 9),
              '12' : (['H'] * 3) + ['F'] + (['H'] * 4) + (['G'] * 2) + (['F'] * 4) + \
                                          ['G'] + (['F'] * 7) + ['G'] + ['H']              
              }

GSCI_comp     = pd.DataFrame(GSCI_comp)
GSCI_products = GSCI_comp['Product'].get_values() 

def roll_spread_months(product, month):
    
    front_month = str(month)
    back_month  = str(1+(month%12))
    GSCI_comp_row = GSCI_comp[GSCI_comp['Product'] == product]

    offset, year_offset = 0,0

    front_month_str = GSCI_comp_row[front_month].get_values()[0]
    back_month_str  = GSCI_comp_row[back_month].get_values()[0]
    
    if contract_month_map[front_month_str] > contract_month_map[back_month_str]:
        offset = 1
    if contract_month_map[front_month_str] < month:
        year_offset = 1

    return front_month_str, back_month_str, offset, year_offset

US_Federal_Calendar = USFederalHolidayCalendar()
bmth_us = CustomBusinessMonthBegin(calendar=US_Federal_Calendar)
bday_us = CustomBusinessDay(calendar=US_Federal_Calendar)

def bus_day_add(dt, num_days):

    if num_days == 0:
        raise RuntimeError('Num of business days specified is 0')
    
    if num_days < 0:
        day_offset  = -2*(abs(num_days)+3)
        num_periods = -1*day_offset 
    else:
        day_offset  = -5
        num_periods = min(5,2*(num_days+3))
        
    dts = [ts.date() for ts in pd.date_range(dt+relativedelta(days=day_offset), periods=num_periods, freq=bday_us)]
    try:
        dt_ind = dts.index(dt)
    except ValueError:
        dt_ind = bisect.bisect_left(dts, dt)
        num_days -= 1

    return dts[dt_ind+num_days]

def roll_dates_by_month(yr, mth):
    yr, mth = int(yr), int(mth)
    dts = [dtime.date() for dtime in pd.date_range(datetime.date(yr, mth, 1) + relativedelta(days=-2) + bmth_us, periods=9, freq=bday_us)]

    return dts[4], dts[-1]


def create_roll_dates(days_offset, roll_date, establish=True):

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

        df_dates = pd.DataFrame( {'Date' : [dt.date() if not type(dt).__name__ == 'date' else dt for dt in _dates] })

    return df_dates

class SummStats(object):
    def __init__(self, price_pos):
        self._price_pos = price_pos

        self.summary_stats()

    def summary_stats(self):
        try:
            self._price_pos['PL'] = np.concatenate([[0], self._price_pos['Position'][:-1] * np.diff(self._price_pos['Settle12'])])
        except TypeError:
            self._price_pos['Settle12'] = self._price_pos['Settle12'].apply(lambda x: float(x))
            self._price_pos['PL'] = np.concatenate([[0], self._price_pos['Price_mult'][:-1] * self._price_pos['Position'][:-1] * np.diff(self._price_pos['Settle12'])])            
    def PL(self):
        return self._price_pos
    
        
    
class Problem_1(object):
    def __init__(self):
        pass

    @staticmethod
    def params():
        return Param(DB=5, acc_method='linear', DA=5, liq_method='linear')

class Problem_2(object):
    def __init__(self):
        pass

    @staticmethod
    def params():
        return Param(DB=10, acc_method='linear', DA=10, liq_method='linear')        

class Problem_3(object):
    def __init__(self):
        pass

    @staticmethod
    def params():
        return Param(DB=2, acc_method='linear', DA=10, liq_method='linear')        

class Visitor(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def visit(self, element):
        pass

class Backtester(Visitor):
    def __init__(self, product, param_obj=Problem_1):
        self._product     = product
        self._exchange    = name_map['exch_map'][self._product]
        self._credentials = Credentials()
        self._metrics_map = {'drawup'  : lib.max_drawup,
                             'drawdn'  : lib.max_drawdown,
                             'sharpe'  : lib.sharpe,
                             'updnrat' : lib.uprat,
                             'freq'    : lib.freq,
                             'ampl'    : lib.ampl}
        self._db_name = '_'.join(['STIR', self._exchange, self._product, 'SUMM'])
        self._SpreadObj = lib.Spread(product)
        self._DBConn = self._DB_conn(*self._credentials.DB_creds(db_override=self._db_name))
        self._params = param_obj.params()

    @staticmethod
    def _DB_conn(user, passwd, db):
        conn = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user,passwd,db))
        if not database_exists(conn.url):
            create_database(conn.url)
        return conn

    def __repr__(self):
        return '{}_{}_{}_{}'.format(
            self._params.DB,
            self._params.acc_method,
            self._params.DA,
            self._params.liq_method
            )

    def visit(self, element):
        pass

    def backtest_helper(self, total_dollars=1e6):

        if self._product not in GSCI_products:
            return None, None, None, None

        portfolio_results = pd.DataFrame()

        RPDW = GSCI_comp[GSCI_comp['Product'] == product]['2017_RPDW'].get_values()[0]

        days_before = self._params.DB
        days_after  = self._params.DA
        est_method  = self._params.acc_method
        liq_method  = self._params.liq_method

        S = lib.Spread(self._product)

        for month_num in range(1,13):

            front_month, back_month, offset, year_offset = roll_spread_months(self._product, month_num)

            if front_month == back_month and offset == 0:
                continue

            price = S._create_study(front_month, back_month, offset, train_only=True)

            all_years = sorted(price.keys())

            for year in all_years:

                curr_year = year
                foll_year = str(int(year)+year_offset)
                if int(foll_year) > max([int(yr) for yr in all_years]):
                    continue

                if not isinstance(price[year], pd.DataFrame):
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

                # Establishing position: calculation of number of units
                est_dates_df  = create_roll_dates(days_before, start_roll_date)

                try:
                    DCRP_ref_date = est_dates_df['Date'].get_values()[0]
                    DCRP = float(price[foll_year][price[foll_year]['Date'] == DCRP_ref_date]['Open1'].get_values()[0])
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
                lots              = round(total_dollars * (RPDW / 100) / float(name_map['lotsize_map'][self._product])
                                          / DCRP / float(name_map['mult_map'][self._product]), 0)
                
                product_dols      = total_dollars * (RPDW / 100)
                dols_per_tick     = float(name_map['lotsize_map'][self._product]) * float(name_map['mult_map'][self._product])

                pos_df = pd.DataFrame({ 'Position' : np.linspace(0, -1*lots, 1+est_dates_df.shape[0])[1:] })
                pos_df = pd.concat([est_dates_df, pos_df], axis=1)
                est_price_pos = pd.merge(pos_df, price[foll_year], how='left')[['Date', 'Prod1', 'Month1', 'Month2', 'Offset',
                                                                                'Settle1', 'Settle2', 'Position', 'Settle12']]
                est_price_pos['Dols']          = product_dols
                est_price_pos['Price_mult']    = dols_per_tick
                est_price_pos['SubStrategy']   = 'EST'
                est_price_pos.rename(columns = { 'Prod1' : 'Prod' }, inplace=True)                

                # Liquidating position
                liq_dates_df = create_roll_dates(days_after, end_roll_date, establish=False)
                pos_df = pd.DataFrame({ 'Position' : np.linspace(1*lots, 0,
                                                                 1+liq_dates_df.shape[0])[:-1] })
                pos_df = pd.concat([liq_dates_df, pos_df], axis=1)
                liq_price_pos = pd.merge(pos_df, price[foll_year],
                                        how='left')[['Date', 'Prod1', 'Month1', 'Month2', 'Offset', 'Settle1', 'Settle2', \
                                                      'Position', 'Settle12']]
                liq_price_pos['Dols']          = product_dols
                liq_price_pos['Price_mult']    = dols_per_tick
                liq_price_pos['SubStrategy']   = 'LIQ'
                liq_price_pos.rename(columns = { 'Prod1' : 'Prod' }, inplace=True)

                SS = SummStats(est_price_pos)
                est_price_pos = SS.PL()

                SS = SummStats(liq_price_pos)
                liq_price_pos = SS.PL()

                portfolio_results = pd.concat([portfolio_results, est_price_pos, liq_price_pos])

        portfolio_results['RPDW'] = RPDW
        portfolio_results['Strategy'] = '_'.join([self._product, self.__repr__()])

        portfolio_results = portfolio_results.sort_values('Date')
        
        summ_table_name = '_'.join(['GSCI_strat_summ', self._product, 'DEFAULT', self.__repr__()])
        portfolio_results.to_sql(con=self._DBConn, name=summ_table_name, if_exists='replace', index=False)

        diffs = pd.Series(np.diff(portfolio_results['PL']))
        metrics  = pd.DataFrame({k:[v(diffs)] for k,v in self._metrics_map.items()})

        metrics_table_name = '_'.join(['GSCI_strat_metrics', self._product, 'DEFAULT', self.__repr__()])
        metrics.to_sql(con=self._DBConn, name=metrics_table_name, if_exists='replace', index=False)
        
        return portfolio_results, metrics, summ_table_name, metrics_table_name
    

if __name__ == '__main__':
    for product in products:
        B1 = Backtester(product, param_obj=Problem_1)
        summ, metrics, summ_table_name, metrics_table_name = B1.backtest_helper()

        B2 = Backtester(product, param_obj=Problem_2)
        summ, metrics, summ_table_name, metrics_table_name = B2.backtest_helper()
        
        B3 = Backtester(product, param_obj=Problem_3)        
        summ, metrics, summ_table_name, metrics_table_name = B3.backtest_helper()

        print('{} FINISHED: WROTE TO TABLE: {}'.format(product, summ_table_name))

