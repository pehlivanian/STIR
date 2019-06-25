import pandas as pd

from db import DBExt
import data

name_map     = data.name_map
train_years  = data.train_years
test_years   = data.test_years
verify_years = data.verify_years

class Element(object):
    def __init__(self):
        pass

    def accept(self, visitor):
        visitor.visit(self)

class Spread(object):
    def __init__(self, prod, db_name=None):
        self._prod = prod
        self._exch = name_map['exch_map'][prod]
        self._DBExt = DBExt(prod, db_name=db_name)

    def _get_spread_keys(self, iden, train_only=False):
        spread_keys = []

        if train_only:
            all_years = train_years
        else:
            all_years = train_years + test_years + verify_years
            
        sk_template = { 'prod1' : iden['prod1'], 'exch1' : iden['exch1'], 'mth1' : iden['front_month'],
                        'prod2' : iden['prod2'], 'exch2' : iden['exch1'], 'mth2' : iden['back_month'],}
        for train_year in all_years:
            spread_key            = sk_template
            spread_key['yr1']     = train_year
            spread_key['yr2']     = str(int(train_year) + iden['offset'])
            spread_key['offset']  = iden['offset']
            spread_keys.append(spread_key.copy())
        
        return spread_keys

    def _get_spread_pair(self, month1, year1, month2, year2):
        df1 = self._DBExt._read_close(month1, year1)
        df2 = self._DBExt._read_close(month2, year2)

        return df1, df2
    
    def _create_spread(self, spread_key):
        ''' Create spread dataframe
        '''

        df1, df2 = self._get_spread_pair(spread_key['mth1'], spread_key['yr1'], spread_key['mth2'], spread_key['yr2'])

        try:
            df = df1.merge(df2, on='Date', how='inner', suffixes=('1','2'),sort=True)
        except AttributeError as e:
            return None

        first_leg       = spread_key['prod1'] + spread_key['mth1'] + spread_key['yr1']
        sec_leg         = spread_key['prod2'] + spread_key['mth2'] + spread_key['yr2']
        
        LTD_FND = self._DBExt._read_LTD_FND()
        LTD      = LTD_FND[LTD_FND['Contract'] == first_leg]['LTD'].values[0]
        FND      = LTD_FND[LTD_FND['Contract'] == first_leg]['FND'].values[0]

        
        BEF_SETTL = 0
        
        df['Days_bef_FND'] = [BEF_SETTL + (x-FND).days for x in df['Date']]
        df = df[df['Days_bef_FND'] < 0]
        df['Norm_day'] = pd.Series(list(reversed(range(df.shape[0]))), \
                                   index=df.index)
        df['Offset'] = [spread_key['offset'] for x in df['Date']]
        
        df['Settle12'] = df['Settle1'] - df['Settle2']
    
        return df
        
    def _create_study(self, front_mth, back_mth, offset, train_only=False):

        LTD_FND = self._DBExt._read_LTD_FND()
        
        r = {}
        iden = {'prod1' : self._prod, 'exch1' : self._exch,
                'prod2' : self._prod, 'exch2' : self._exch,
                'front_month' : front_mth, 'back_month' : back_mth, 'offset' : offset }

        spread_keys = self._get_spread_keys(iden, train_only=train_only)

        for spread_key in spread_keys:
            spread_df = self._create_spread(spread_key)
            r[spread_key['yr1']] = spread_df
            
        return r

