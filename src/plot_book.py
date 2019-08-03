import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations, permutations
from matplotlib.backends.backend_pdf import PdfPages

import lib
import db
from data import (products, nrbys, months, years, fields, name_map, train_years, test_years, verify_years)

def read_GSCI_summary_tables(product):

    exchange = name_map['exch_map'][product]
    db_name = '_'.join(['STIR', exchange, product, 'SUMM'])
    db_conn = db.DBExt(product, db_name=db_name)
    tables = db_conn._metadata.tables

    rpts = list()
    metrics = pd.DataFrame()
    
    for k,v in tables.items():
        if k.startswith('GSCI_strat_summ'):
            table_data = v.select().execute().fetchall()
            columns    = [column.name for column in v.columns]
            df = pd.DataFrame(table_data, columns=columns)
            rpts.append(df)
        if k.startswith('GSCI_strat_metrics'):
            table_data = [x if x else 0 for x in v.select().execute().fetchall()[0]]
            columns    = [column.name for column in v.columns]
            df = pd.DataFrame(dict(zip(columns, table_data)), index=[0])
            df['Strat'] = '_'.join(k.split('_')[4:])
            metrics = pd.concat([metrics, df], ignore_index=True)

    if metrics.shape[0]:
        metrics = metrics.set_index('Strat', drop=True)
        metrics = metrics.astype('float')
            
    return rpts, metrics

def plot_GSCI_book_combined(filename='./figs/GSCI_spreads_combined.pdf'):

    def prot_sum( a, b):
        return np.nansum( [ a, b ] )
    
    from collections import defaultdict
    select_cols = [ 'Position', 'Dols', 'PL' ]
    groupby_cols = [ 'Date' ]
    cols = select_cols + groupby_cols
    suffixes = ['_base', '_new' ]    
    reports_all = defaultdict(lambda *a: pd.DataFrame(columns=cols))
    
    with PdfPages(filename) as pdf:
        for product in products:
            reports_by_sector = defaultdict(lambda: defaultdict(list))
            reports,metrics = read_GSCI_summary_tables(product)

            exchange =  name_map['exch_map'][product]
            sector = name_map['sector_map'][product]
            
            for ind,report in enumerate(reports):
                strat = metrics.index[ind].split(product)[-1].split(exchange+'_')[-1]
                report['Position'] = report['Position'].astype(np.float)
                report['PL']       = report['PL'].astype(np.float)
                reports_all[strat] = pd.concat([reports_all[strat], report], axis=0)
                
            base_title = product + ' ' + exchange + ' : ' + sector

            if reports:
                title = base_title
                fig,_ = lib.plot_GSCI_summ_all(reports, title=title)
                pdf.savefig(fig)

                for strategy in ['EST', 'LIQ']:
                    title = base_title + '_by_strategy: {}'.format(strategy)
                    fig,_ = lib.plot_GSCI_summ_bystrat_all(reports, strategy, title=title)
                    pdf.savefig(fig)

            if metrics.shape[0]:
                fig = plot_GSCI_metrics_by_product(metrics, product)
                pdf.savefig(fig)

            print('{}'.format(product))

        combined_report = {k:reports_all[k].groupby(by=['Date'],as_index=False)[['Dols', 'PL']].sum() for k in reports_all.keys()}

        report_list = list()
        for strat,report in combined_report.items():
            report['Strategy'] = strat
            report_list.append(report)
        fig,_ = lib.plot_GSCI_summ_all( report_list, title='Cumulative Strategy')
            
        pdf.savefig(fig)
    
def plot_GSCI_book(filename='./figs/GSCI_spreads.pdf'):
    reports_by_sector = defaultdict(lambda: defaultdict(list))
    reports, metrics = read_GSCI_summary_tables(product)
    
    with PdfPages(filename) as pdf:
        for product in products:
            # Create the sector-grouped reports along the way
            sector = name_map['sector_map'][product]
            for report in reports:
                # strategy_short_old = '_'.join(report['Strategy'].get_values()[0].split('_')[1:])
                strategy_short = report['Strategy'].get_values()[0]
                reports_by_sector[strategy_short][sector].append(report)
                
            rpt_figs = plot_GSCI_summ_by_product(reports)
            for fig in rpt_figs:
                pdf.savefig(fig)

            print('{}'.format(product))

            if metrics.shape[0]:
                met_fig = plot_GSCI_metrics_by_product(metrics, product)
                pdf.savefig(met_fig)

        # for strategy,sector_reports in reports_by_sector.items():
        #     for sector,reports in sector_reports.items():
        #         import pdb
        #         pdb.set_trace()
        #         print('BOMB')
                    
                    

def plot_GSCI_summ_by_product_combined(reports):
    lib.plot_GSCI_summ_all(reports)
        
def plot_GSCI_summ_by_product(reports):

    figs = list()
    for report in reports:
        fig = lib.plot_GSCI_summ(report)
        figs.append(fig)
    return figs

def plot_GSCI_metrics_by_product(metrics, product):

    fig = lib.plot_GSCI_metrics(metrics, product)

    return fig
    
def plot_GSCI_by_sector():
    pass
    
def plot_book(products=products):
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

# if __name__ == '__main__':
#     # plot_book()
#     # plot_GSCI_book()
