import pandas as pd
from collections import defaultdict
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

def read_GSCI_summary_tables(product):

    exchange = name_map['exch_map'][product]
    db_name = '_'.join(['STIR', exchange, product, 'SUMM'])
    db_conn = db.DBExt(product, db_name=db_name)
    tables = db_conn._metadata.tables

    rpts = list()
    for k,v in tables.items():
        if k.startswith('GSCI_strat_summ'):
            table_data = v.select().execute().fetchall()
            columns    = [column.name for column in v.columns]
            df = pd.DataFrame(table_data, columns=columns)
            rpts.append(df)

    return rpts
    
def plot_GSCI_book(filename='./figs/GSCI_spreads.pdf'):

    reports_by_sector = defaultdict(lambda: defaultdict(list))
    
    with PdfPages(filename) as pdf:
        for product in products:
            reports = read_GSCI_summary_tables(product)

            # Create the sector-grouped reports along the way
            sector = name_map['sector_map'][product]
            for report in reports:
                strategy_short = '_'.join(report['Strategy'].get_values()[0].split('_')[1:])
                reports_by_sector[strategy_short][sector].append(report)
                
            figs = plot_GSCI_by_product(reports)
            for fig in figs:
                pdf.savefig(fig)

        # for strategy,sector_reports in reports_by_sector.items():
        #     for sector,reports in sector_reports.items():
        #         import pdb
        #         pdb.set_trace()
        #         print('BOMB')
                    
                    
               
def plot_GSCI_by_product(reports):

    figs = list()
    for report in reports:
        fig = lib.plot_GSCI(report)
        figs.append(fig)
    return figs
    
def plot_GSCI_by_sector():
    pass
    
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
