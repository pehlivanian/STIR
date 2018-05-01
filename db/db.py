# Calling convention
# to read tables in pyspark:
# ~/spark-1.6.1/bin/pyspark --jars /home/charles/spark-1.6.1/mysql-connector-java-5.1.39/mysql-connector-java-5.1.39-bin.jar 

import logging
import pandas as pd
import numpy as np
import quandl
import string
import datetime as dt
import MySQLdb as mariadb
from pymongo import MongoClient
from sqlalchemy import create_engine, Table, Column, MetaData
from sqlalchemy_utils import database_exists, create_database
from contextlib import contextmanager

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


logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)


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
                            

