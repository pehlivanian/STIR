import pandas as pd
from itertools import combinations, permutations
from matplotlib.backends.backend_pdf import PdfPages

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

def plot_GSCI_book():

    # XXX
    products = ['NG']
    
    # Results by product
    for product in products:
        exchange = name_map['exch_map'][product]
        db_name = '_'.join(['STIR', exchange, product, 'SUMM'])
        db_conn = db.DBExt(product, db_name=db_name)
        tables = db_conn._metadata.tables
        for k,v in tables.items():
            table_data = v.select().execute().fetchall()
            columns    = [column.name for column in v.columns]
            df = pd.DataFrame(table_data, columns=columns)

            fig = lib.plot_GSCI(df)




def plot_book():
    for product in products:
        plot_book_by_product(product, max_offset=2, filename='./figs/all_{}_spreads.pdf'.format(product))        

def plot_book_by_product(product, max_offset=2, filename='./all_spreads.pdf'):

    SpreadObj = lib.Spread(product)

    combs = list(combinations(months, 2))

    with PdfPages(filename) as pdf:
        for (front_mth,back_mth) in permutations(months, 2):

            if (front_mth,back_mth) in combs:
                for offset in range(0, 1+max_offset):

                    try:
                        price = SpreadObj._create_study(front_mth, back_mth, offset)
                        fig   = lib.plot_spreads(price)
                        pdf.savefig(fig)
                        print('Included  spread: {}:{}-{} offset: {}'.format(product, front_mth, back_mth, offset))                                                
                    except Exception as e:
                        continue
            else:
                for offset in range(1, 1+max_offset):

                    try:
                        price = SpreadObj._create_study(front_mth, back_mth, offset)
                        fig   = lib.plot_spreads(price)
                        pdf.savefig(fig)
                        print('Included  spread: {}:{}-{} offset: {}'.format(product, front_mth, back_mth, offset))                        

                    except Exception as e:
                        continue

if __name__ == '__main__':
    # plot_book()
    plot_GSCI_book()
