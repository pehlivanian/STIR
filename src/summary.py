import pandas as pd
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

def GSCI_summary():

    metrics_map = {'drawupK'   : partial(lib.max_drawup_levels,   mult=1e-3),
                   'drawdnK'   : partial(lib.max_drawdown_levels, mult=1e-3),
                   'meanretK'  : partial(lib.meanret_levels,      mult=1e-3),
                   'sharpe'    : lib.sharpe_levels,
                   'updnrat'   : lib.uprat_levels,
                   'nonnegrat' : lib.nonnegrat_levels,
                   'freq'      : lib.freq_levels,
                   'ampl'      : lib.ampl_levels}
    
    def merge_sum(df1, df2):
        df = pd.DataFrame(pd.merge(df1[['Date', 'PL']], df2[['Date', 'PL']], on='Date', how='outer', suffixes=['1','2']))
        df = df.fillna(0)
        df = pd.DataFrame(df.set_index('Date').sum(axis=1), columns=['PL'])
        df = df.reset_index(level=0)
        return df

    SUMM_COLS = ['Date', 'Prod', 'PL']
    
    PL_summary = pd.DataFrame(columns=['PL', 'Dols'])
    reports_by_sector = defaultdict(lambda: defaultdict(pd.DataFrame))
    PL_by_sector      = defaultdict(lambda: defaultdict(pd.DataFrame))
    
    for product in products:
        reports, metrics = read_GSCI_summary_tables(product)

        # Create the sector-grouped reports along the way
        sector = name_map['sector_map'][product]
        for report in reports:
            strategy_short = '_'.join(report['Strategy'].get_values()[0].split('_')[1:])
            report_short = report[SUMM_COLS]
            report_short['MaxLots'] = max(report['Position'])
            report_short['MaxDols'] = max(report['Dols'])
            reports_by_sector[strategy_short][sector] = pd.concat([reports_by_sector[strategy_short][sector], report_short])

            report['PL'] = report['PL'].astype('float')
            if PL_by_sector[strategy_short][sector].shape[0] == 0:
                PL_by_sector[strategy_short][sector] = report[['Date', 'PL']]
            else:
                PL_by_sector[strategy_short][sector] = merge_sum(PL_by_sector[strategy_short][sector], report[['Date', 'PL']])
                        
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
            
            

        
