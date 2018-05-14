import pandas as pd
from functools import partial
from collections import defaultdict
from itertools import combinations, permutations

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

def merge_sum(df1, df2):
    if df1.shape[0] == 0:
        return df2
    if df2.shape[0] == 0:
        return df1
    
    df = pd.DataFrame(pd.merge(df1[['Date', 'PL']], df2[['Date', 'PL']], on='Date', how='outer', suffixes=['1','2']))
    df = df.fillna(0)
    df = pd.DataFrame(df.set_index('Date').sum(axis=1), columns=['PL'])
    df = df.reset_index(level=0)
    return df


def GSCI_summary_by_prod(substrategy='All'):

    COLS = ['Date', 'Prod', 'Position', 'Dols', 'Settle12', 'PL']
    metrics_map = {'drawupK'   : partial(lib.max_drawup_levels,   mult=1e-3),
                   'drawdnK'   : partial(lib.max_drawdown_levels, mult=1e-3),
                   'meanretK'  : partial(lib.meanret_levels,      mult=1e-3),
                   'sharpe'    : lib.sharpe_levels,
                   'updnrat'   : lib.uprat_levels,
                   'nonnegrat' : lib.nonnegrat_levels,
                   'freq'      : lib.freq_levels,
                   'ampl'      : lib.ampl_levels}


    # Create monolithic DataFrame?

    reports_all = pd.DataFrame()
    for product in products:
        sector = name_map['sector_map'][product]        
        reports, metrics = read_GSCI_summary_tables(product)

        for report in reports:
            report['Sector'] = sector
            reports_all = pd.concat([reports_all, report])

    # Summary by product
    groups = reports_all.groupby(['Strategy', 'Sector', 'Prod'])
    summ_by_prod = groups['PL'].agg([(k,v) for k,v in metrics_map.items()])

    return summ_by_prod
            
            
def GSCI_summary(substrategy='All'):

    metrics_map = {'drawupK'   : partial(lib.max_drawup_levels,   mult=1e-3),
                   'drawdnK'   : partial(lib.max_drawdown_levels, mult=1e-3),
                   'meanretK'  : partial(lib.meanret_levels,      mult=1e-3),
                   'sharpe'    : lib.sharpe_levels,
                   'updnrat'   : lib.uprat_levels,
                   'nonnegrat' : lib.nonnegrat_levels,
                   'freq'      : lib.freq_levels,
                   'ampl'      : lib.ampl_levels}
    
    SUMM_COLS = ['Date', 'Prod', 'PL']
    
    PL_summary = pd.DataFrame(columns=['PL', 'Dols'])
    reports_by_sector = defaultdict(lambda: defaultdict(pd.DataFrame))
    PL_by_sector      = defaultdict(lambda: defaultdict(pd.DataFrame))
    
    for product in products:
        reports, metrics = read_GSCI_summary_tables(product)

        if substrategy   == 'EST':
            reports = [report[report['SubStrategy']=='EST'] for report in reports]
        elif substrategy == 'LIQ':
            reports = [report[report['SubStrategy']=='LIQ'] for report in reports]

        # Create the sector-grouped reports along the way
        sector = name_map['sector_map'][product]
        for report in reports:
            strategy = report['Strategy'].get_values()[0]
            report_short = report[SUMM_COLS]
            report_short['MaxLots'] = max(report['Position'])
            report_short['MaxDols'] = max(report['Dols'])
            reports_by_sector[strategy][sector] = pd.concat([reports_by_sector[strategy][sector], report_short])

            report['PL'] = report['PL'].astype('float')
            PL_by_sector[strategy][sector] = merge_sum(PL_by_sector[strategy][sector], report[['Date', 'PL']])
                        
    sectors = list(reports_by_sector['5_linear_5_linear'].keys())
    metrics_by_prod = pd.DataFrame()

    for sector in sectors:
        groups = reports_by_sector['5_linear_5_linear'][sector].groupby(['Prod'])
        for prod,group in groups:
            levels = pd.Series(group['PL']).astype('float')
            m = pd.DataFrame({k:[v(levels)] for k,v in metrics_map.items()})
            m['Sector'] = sector
            m['Prod'] = prod
            m['MaxLots'] = group['MaxLots'].get_values()[0]
            m['MaxDols'] = group['MaxDols'].get_values()[0]
            metrics_by_prod = pd.concat([metrics_by_prod, m])


    sectors = list(reports_by_sector['5_linear_5_linear'].keys())
    metrics_by_sector = pd.DataFrame()

    for sector in sectors:
        df = PL_by_sector['5_linear_5_linear'][sector]
        levels = pd.Series(df['PL']).astype('float')
        m = pd.DataFrame({k:[v(levels)] for k,v in metrics_map.items()})
        m['Sector'] = sector
        metrics_by_sector = pd.concat([metrics_by_sector, m])

    return metrics_by_prod, metrics_by_sector


def unpickle():
    WORKING_DIR = '/home/charles/dev/Python/STIR/'

    global prod_ALL_train,   prod_EST_train,   prod_LIQ_train
    global prod_ALL_all,     prod_EST_all,     prod_LIQ_all
    global sector_ALL_train, sector_EST_train, sector_LIQ_train
    global sector_ALL_all,   sector_EST_all,   sector_LIQ_all

    
    prod_ALL_train = pd.read_pickle(WORKING_DIR + 'figs/prod_ALL_train.pkl')
    prod_EST_train = pd.read_pickle(WORKING_DIR + 'figs/prod_EST_train.pkl')
    prod_LIQ_train = pd.read_pickle(WORKING_DIR + 'figs/prod_LIQ_train.pkl')

    sector_ALL_train = pd.read_pickle(WORKING_DIR + 'figs/sector_ALL_train.pkl')
    sector_EST_train = pd.read_pickle(WORKING_DIR + 'figs/sector_EST_train.pkl')
    sector_LIQ_train = pd.read_pickle(WORKING_DIR + 'figs/sector_LIQ_train.pkl')

    prod_ALL_all = pd.read_pickle(WORKING_DIR + 'figs/prod_ALL_all.pkl')
    prod_EST_all = pd.read_pickle(WORKING_DIR + 'figs/prod_EST_all.pkl')
    prod_LIQ_all = pd.read_pickle(WORKING_DIR + 'figs/prod_LIQ_all.pkl')

    sector_ALL_all = pd.read_pickle(WORKING_DIR + 'figs/sector_ALL_all.pkl')
    sector_EST_all = pd.read_pickle(WORKING_DIR + 'figs/sector_EST_all.pkl')
    sector_LIQ_all = pd.read_pickle(WORKING_DIR + 'figs/sector_LIQ_all.pkl')
    
if __name__ == '__main__':

    WORKING_DIR = '/home/charles/dev/Python/STIR/'
    prod_ALL, sector_ALL = GSCI_summary(substrategy='ALL')

    prod_ALL.to_pickle(WORKING_DIR + 'figs/prod_ALL_all.pkl')
    sector_ALL.to_pickle(WORKING_DIR + 'figs/sector_ALL_all.pkl')    
    
    prod_EST, sector_EST = GSCI_summary(substrategy='EST')

    prod_EST.to_pickle(WORKING_DIR + 'figs/prod_EST_all.pkl')
    sector_EST.to_pickle(WORKING_DIR + 'figs/sector_EST_all.pkl')    

    prod_LIQ, sector_LIQ = GSCI_summary(substrategy='LIQ')

    prod_LIQ.to_pickle(WORKING_DIR + 'figs/prod_LIQ_all.pkl')
    sector_LIQ.to_pickle(WORKING_DIR + 'figs/sector_LIQ_all.pkl')    
    
            
            

        
