import data as config_data
import pandas as pd

import MySQLdb as mariadb
from pymongo import MongoClient
from sqlalchemy import create_engine, Table, Column, MetaData, inspect
from sqlalchemy_utils import database_exists, create_database
from contextlib import contextmanager

products     = config_data.all_products()
nrbys        = config_data.all_nrbys()
months       = config_data.all_months()
years        = config_data.all_years()
fields       = config_data.all_fields()
name_map     = config_data.all_names()
train_years  = config_data.train_years()
test_years   = config_data.test_years()
verify_years = config_data.verify_years()

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

                            
                            
            
    

