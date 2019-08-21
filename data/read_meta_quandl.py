import string

# working_dir = "/home/charles/git/STIR_Python/data/"
# working_dir = "/home/charles/git/STIR_ApacheSpark/data/"
working_dir = '/home/charles/dev/Python/STIR/data/'

names_dir = working_dir + "ALL_NAMES/"
years_dir = working_dir + "ALL_DATES/"
months_dir = working_dir + "ALL_DATES/"
fields_dir = working_dir + "ALL_NAMES/"
nrby_dir = working_dir + "ALL_DATES/"

def all_fields():
    fn = fields_dir + "all_fields.dat"
    
    r = []
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        r.append(line.rstrip())
        
    return r

def all_months():
    fn = months_dir + "all_months.dat"
    
    r = []
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        r.append(line.rstrip())
        
    return r

def all_years():
    fn = years_dir + "all_years.dat"
    
    r = []
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        r.append(line.rstrip())
        
    return r

def train_years():
    fn = years_dir + "train_years.dat"

    r = []
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        r.append(line.rstrip())

    return r

def test_years():
    fn = years_dir + "test_years.dat"

    r = []
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        r.append(line.rstrip())

    return r

def verify_years():
    fn = years_dir + "verify_years.dat"

    r = []
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        r.append(line.rstrip())

    return r

def all_letters():
    r = ("F","G","H",
    "J","K","M",
    "N","Q","U",
    "V","X","Z")
    return r
         
def all_nrbys():
    fn = nrby_dir + "all_nrbys.dat"
    
    r = []
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        r.append(line.rstrip())
        
    return r

def all_names():
    sym = []
    ticker = []
    lotsize = []
    mult = []
    exch = []
    q_str = []
    sector = []

    fn = names_dir + "names_tickers.dat"
    file = open(fn)
    lines = file.readlines()
    for line in lines:
        line_sp = line.split(',')
        sym.append(line_sp[0])
        ticker.append(line_sp[1])
        lotsize.append(line_sp[2])
        mult.append(line_sp[3])
        exch.append(line_sp[4])
        q_str.append(line_sp[5])
        sector.append(line_sp[6].rstrip())

    ticker_map = dict(zip(sym,ticker))
    lotsize_map = dict(zip(sym,lotsize))
    mult_map = dict(zip(sym,mult))
    exch_map = dict(zip(sym,exch))
    q_str_map = dict(zip(sym,q_str))
    sector_map = dict(zip(sym,sector))
    
    
    return {'ticker_map' : ticker_map,
            'lotsize_map': lotsize_map,
            'mult_map'   : mult_map,
            'exch_map'   : exch_map,
            'q_str_map'  : q_str_map,
            'sector_map' : sector_map,
            }
    
def all_products():
    return all_names()['ticker_map'].keys()
