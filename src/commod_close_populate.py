# Calling convention
# to read tables in pyspark:
# ~/spark-1.6.1/bin/pyspark --jars /home/charles/spark-1.6.1/mysql-connector-java-5.1.39/mysql-connector-java-5.1.39-bin.jar 

import logging
from data import products
import db

def populate_data(products=products):

    for product in products:

        print('PRODUCT: {!r}'.format(product))
        db_inst = db.DBExt(product)
        
        db_inst._commit_close()
        print('FINISHED PERSISTING CLOSE TABLE FOR {!r}'.format(product))

        db_inst._commit_LTD_FND()
        print('FINISHED PERSISTING LTD_FND TABLE  FOR {!r}'.format(product))
        
