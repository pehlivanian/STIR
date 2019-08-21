# Calling convention
# to read tables in pyspark:
# ~/spark-1.6.1/bin/pyspark --jars /home/charles/spark-1.6.1/mysql-connector-java-5.1.39/mysql-connector-java-5.1.39-bin.jar 

import logging
import pandas as pd
import numpy as np
import quandl
import string
from datetime import timedelta
import datetime
import MySQLdb as mariadb
from pymongo import MongoClient
from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy_utils import database_exists, create_database
from contextlib import contextmanager

from data import *

# Duplicated from utils
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import CustomBusinessMonthBegin

US_Federal_Calendar = USFederalHolidayCalendar()
bmth_us = CustomBusinessMonthBegin(calendar=US_Federal_Calendar)
bday_us = CustomBusinessDay(calendar=US_Federal_Calendar)

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)


Dtype_Mapping = {
    'object' : 'TEXT',
    'int64'  : 'INT',
    'float64' : 'FLOAT',
    'datetime64' : 'DATETIME',
    'bool' : 'TINYINT',
    'category' : 'TEXT',
    'timedelta[ns]' : 'TEXT'
    }

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class QuandlExt(object):
    def __init__(self):
        self._credentials = Credentials()

    @contextmanager
    def open(self):
        quandl.ApiConfig.api_key = self._credentials.quandl_creds()
        yield quandl
        logging.info('Releasing quandl contextmanager')

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

class DBExt(object):
    def __init__(self, prod, db_name=None):
        self._prod        = prod
        self._exch        = name_map['exch_map'][prod]
        self._credentials = Credentials()
        self._quandl      = QuandlExt()

        self._db_name = db_name
        if not db_name:
            self._db_name = '_'.join(['STIR', self._exch, self._prod])

        self._conn        = self._DB_conn(*self._credentials.DB_creds(db_override=self._db_name))
        self._metadata    = MetaData(self._conn, reflect=True)

    @staticmethod
    def _DB_conn(user, passwd, db):
        conn = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user,passwd,db))
        if not database_exists(conn.url):
            create_database(conn.url)
        return conn

    def table_names(self):
        return self._conn.table_names()

    def _contract_name(self, month, year):
        return name_map['q_str_map'][self._prod]+ month + year

    def _table_name(self, month, year):
        table_name = "_".join([self._exch, name_map['ticker_map'][self._prod]])
        return table_name + month + year

    def _LTD_FND_table_name(self):
        table_name = '_'.join([self._exch, self._prod, 'LTD', 'FND'])
        return table_name

    def _read_close(self, month, year):

        table_name     = self._table_name(month, year)
        self._metadata = MetaData(self._conn, reflect=True)
        
        try:
            table = self._metadata.tables[table_name]
        except KeyError:
            logging.warn('No such table in db: {!r}'.format(table_name))
            return None

        data = table.select().execute().fetchall()
        columns = [c.description for c in table.columns]

        df = pd.DataFrame(data, columns=columns)
        df['Prod']  = self._prod
        df['Exch']  = self._exch
        df['Year']  = int(year)
        df['Month'] = month

        return df

    def _read_LTD_FND(self):
        table_name = self._LTD_FND_table_name()
        try:
            table = self._metadata.tables[table_name]
        except KeyError:
            logging.error('No such tabke in db: {!r}'.format(table_name))
            return None

        data = table.select().execute().fetchall()
        columns = [c.description for c in table.columns]

        df = pd.DataFrame(data, columns=columns)
        return df

    @staticmethod
    def generate_trade_dates(date1, date2):
        ''' Includes endpoints
        '''
        return [ts.date() for ts in pd.date_range(date1, periods=(date2-date1).days,
                                                  freq=bday_us) if ts.date() <= date2]

    @staticmethod
    def _dt_to_dttm(dt,day_offset=0):
        return datetime.datetime.combine(dt+timedelta(day_offset), datetime.datetime.min.time())

    @staticmethod
    def _dttm_to_dt(dttm, day_offset=0):
        return dttm.date()+timedelta(day_offset)
                
    def _commit_continuous_contract(self, max_contract_num=5, days_before_last=5, method='linear'):

        LTD_FND = self._read_LTD_FND()
        LTD_FND['LTD'] = LTD_FND.apply(lambda row:  self._dt_to_dttm(row.LTD,0), axis=1)
        LTD_FND['FND'] = LTD_FND.apply(lambda row: self._dt_to_dttm(row.FND,0), axis=1)
        LTD_FND['mindate'] = LTD_FND.apply(lambda row: min(row.LTD, row.FND), axis=1)

        cc_info = pd.DataFrame( { 'date' : LTD_FND['mindate'] } )
        cc_info['contract_0'] = LTD_FND['Contract']
                
        for offset in range(max_contract_num):
            mindate_col = 'mindate' + str(offset)
            src_contract_col = 'Contract_next'
            dest_contract_col = 'contract_'+str(offset+1)
            LTD_FND[mindate_col] = LTD_FND['mindate'] + timedelta(1)
            select_cols = ['Contract', 'LTD', 'FND', 'mindate', mindate_col]
            merge = pd.merge_asof(LTD_FND[select_cols],
                                  LTD_FND[select_cols],
                                  left_on=mindate_col,
                                  right_on='mindate',
                                  direction='forward',
                                  suffixes=['', '_next']
                                     )
            cc_info[dest_contract_col] = merge[src_contract_col]
            LTD_FND['Contract'] = pd.merge(LTD_FND[['LTD', 'FND']],
                                           merge[['LTD', 'FND', 'Contract_next']],
                                           how='left'
                                           )['Contract_next']
        
        # Form the settles dataframe
        first_date = datetime.date(year=int(years[0]),
                                   month=datetime.date.min.month,
                                   day=datetime.date.min.day
                                   )
        last_date = datetime.date(year=int(years[-1])+1,
                                   month=datetime.date.min.month,
                                   day=datetime.date.min.day
                                   )

        # Form dataframe with trade dates
        trade_dates = self.generate_trade_dates(first_date, last_date)
        cont_settle = pd.DataFrame()
        cont_settle['date'] = trade_dates
        cont_settle['date'] = cont_settle.apply(lambda row: self._dt_to_dttm(row.date), axis=1)

        # Fill in contract info
        cont_settle = pd.merge_asof(cont_settle, cc_info, on='date', direction='forward')

        # Gather contracts needed
        contracts = set()
        for _,row in cont_settle.iterrows():
            for num in range(max_contract_num+1):
                contract_name = row.get('contract_'+str(num))
                if isinstance(contract_name, str):
                    contracts.add(row.get('contract_'+str(num)))

        # Gather settles time series
        contracts_ts = pd.DataFrame()
        for contract in contracts:
            month = contract[-5:-4]
            year = contract[-4:]
            contract_ts = self._read_close(month, year)
            contract_ts['date'] = contract_ts.apply( lambda row: self._dt_to_dttm(row.Date), axis=1)
            contract_ts['contract'] = contract
            contracts_ts = pd.concat([contracts_ts, contract_ts])
        contracts_ts = contracts_ts.sort_values(by=['date'])

        # Populate cont_settles with settle information
        base_cols = ['date', 'contract']
        price_cols = ['Settle', 'Volume', 'Open Interest', 'Low', 'High']
        ts_cols = base_cols + price_cols
        for num in range(max_contract_num+1):
            contract_col = 'contract_'+str(num)
            cont_settle = pd.merge_asof(cont_settle,
                                        contracts_ts[ts_cols],
                                        on='date',
                                        left_by=contract_col,
                                        right_by='contract',
                                        )
            cont_settle = cont_settle.rename(columns={col:'_'.join([col.lower(), str(num)]) for col in price_cols})
            cont_settle = cont_settle.drop(columns=['contract'])

        cont_settle_all = cont_settle.copy()
        if method == 'linear':
            for num in range(max_contract_num):
                cont_settle = cont_settle_all
                # Set blend weights            
                front_weight_col = 'weight_0'
                back_weight_col = 'weight_1'
                front_contract_col = 'contract_'+str(num)
                back_contract_col = 'contract_'+str(num+1)
                cont_settle['front_diff'] = cont_settle[front_contract_col].ne(cont_settle[front_contract_col].shift(-days_before_last).bfill())
                def _group_apply(df, front_weight_col, back_weight_col, contract_num):
                    df[front_weight_col] = 1. - (df['front_diff'].cumsum()/contract_num)
                    df[back_weight_col] = 1. - df[front_weight_col]
                    return df
                cont_settle = cont_settle.groupby(by=front_contract_col, as_index=False).apply(_group_apply,
                                                                                       front_weight_col,
                                                                                       back_weight_col,
                                                                                       max_contract_num)

                # Form blended series
                cont_settle = cont_settle[['date', front_contract_col, back_contract_col] +
                                          [front_weight_col, back_weight_col] +
                                          ['_'.join([col.lower(), str(num)]) for col in price_cols] +
                                          ['_'.join([col.lower(), str(num+1)]) for col in price_cols]]
                for col in price_cols:
                    dest_col = col.lower()
                    front_source_col = '_'.join([col.lower(), str(num)])
                    back_source_col = '_'.join([col.lower(), str(num+1)])
                    cont_settle[front_source_col] = cont_settle[front_source_col].astype('float')
                    cont_settle[back_source_col] = cont_settle[back_source_col].astype('float')                
                    cont_settle[dest_col] = (cont_settle[front_weight_col] *
                                             cont_settle[front_source_col]) + (cont_settle[back_weight_col] *
                                                                               cont_settle[back_source_col])
                    cont_settle = cont_settle.drop(columns=[front_source_col, back_source_col])
                cont_settle = cont_settle.rename(columns={front_contract_col: 'contract0', back_contract_col: 'contract1',
                                                front_weight_col: 'weight0', back_weight_col: 'weight1'})



                    # commit
                table_name = '_'.join([self._exch, self._prod, str(num), 'cont'.upper()])
                cont_settle.to_sql(con=self._conn, name=table_name, if_exists='replace', index=False)
                
        
    def _commit_close(self):
        for year in years:
            for month in months:
                contract = self._contract_name(month, year)
                logging.info('Sourcing contract: {}'.format(contract))
                table_name = self._table_name(month, year)
                try:
                    with self._quandl.open() as q:
                        price_vol_data = q.get(contract, returns='pandas')
                except Exception as e:
                    logging.info("--> No data for contract: " + contract + " <--")
                    continue
                price_vol_data.reset_index(level=0, inplace=True)
                price_vol_data['Date'] = price_vol_data['Date'].apply(lambda x: x.date())

                new_column_names = dict(zip(price_vol_data.columns,
                                            [x.replace(" ","_").replace(".","") for x in list(price_vol_data.columns)]))

                price_vol_data.rename(columns=new_column_names)

                price_vol_data.to_sql(con=self._conn, name=table_name, if_exists='replace', index=False)
                logging.info('Successfully wrote to: {!r}'.format(table_name))
                
    def _commit_LTD_FND(self):
        LTD_FND_all = pd.DataFrame()
        
        for year in years:
            LTD, FND = {},{}
            for month in months:
                try:
                    df = self._read_close(month, year)
                    k = str(df['Prod'][0] + df['Month'][0] + str(df['Year'][0]))
                    # XXX
                    # LTD == FND for now
                    LTD[k] = [pd.to_datetime(str(df[-1:]['Date'].values[0])).date()]
                    FND[k] = [pd.to_datetime(str(df[-1:]['Date'].values[0])).date()]
                except Exception as e:
                    continue
            LTD = pd.melt(pd.DataFrame(LTD), var_name='Contract', value_name='LTD')
            FND = pd.melt(pd.DataFrame(FND), var_name='Contract', value_name='FND')
            LTD_FND = LTD.merge(FND, on='Contract')        

            
            LTD_FND_all = pd.concat([LTD_FND_all, LTD_FND])
            

        table_name = "{}_{}_LTD_FND".format(self._exch, self._prod)
        LTD_FND_all.to_sql(con=self._conn, name=table_name, if_exists='replace', index=False)
        logging.info('Successfully wrote to: '.format(table_name))
                            

