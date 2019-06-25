from .read_meta_quandl import ( all_products, all_nrbys, all_months, all_years, all_fields, all_names, train_years, test_years, verify_years )
from .GSCI import GSCIData

products     = all_products()
nrbys        = all_nrbys()
months       = all_months()
years        = all_years()
fields       = all_fields()
name_map     = all_names()
train_years  = train_years()
test_years   = test_years()
verify_years = verify_years()

