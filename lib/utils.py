import pandas as pd
from collections import defaultdict

import MySQLdb as mariadb
from pymongo import MongoClient
from sqlalchemy import create_engine, Table, Column, MetaData, inspect
from sqlalchemy_utils import database_exists, create_database
from contextlib import contextmanager

import sys
sys.path.append('../')

import db
import data

products     = data.all_products
nrbys        = data.all_nrbys
months       = data.all_months
years        = data.all_years
fields       = data.all_fields
name_map     = data.all_names
train_years  = data.train_years
test_years   = data.test_years
verify_years = data.verify_years

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

def list_databases():
    credentials = Credentials()
    user, passwd, db = credentials.DB_creds()
    conn = create_engine('mysql+mysqldb://{}:{}@localhost'.format(user,passwd))
    insp = inspect(conn)
    
    return insp.get_schema_names()

def list_tables(db_name):
    credentials = Credentials()
    user, passwd, db = credentials.DB_creds(db_override=db_name)
    conn = create_engine('mysql+mysqldb://{}:{}@localhost/{}'.format(user, passwd, db_name))
    return conn.table_names()
    
def check_close_tables(summ_only=False):
    db_length, db_name = list(), list()
    
    dbs = list_databases()
    for db in dbs:
        if (db.startswith('STIR') and not db.endswith('SUMM') and not summ_only) or (summ_only and db.startswith('STIR') and db.endswith('SUMM')):
            num_tables = len(list_tables(db))
            db_name.append(db)
            db_length.append(num_tables)

    return pd.DataFrame( { 'Table' : db_name, 'Length' : db_length })

def get_summary(products):

    summ_dataframes = defaultdict(list)
    
    exchange = name_map['exch_map'][product]
    db_name  = '_'.join(['STIR', exchange, product, 'SUMM'])
    db_conn  = db.DBExt(product, db_name=db_name)
    tables   = db_conn._metadata.tables

    for k,v in tables.items():
        table_data = v.select().execute().fetchall()
        columns    = [column.name for column in v.columns]
        df         = pd.DataFrame(table_data, columns=columns)

        df_key = k.split(product)[-1][1:]
        summ_dataframes[df_key].append(df)

    return summ_dataframes
        


                            
                            
            
    

