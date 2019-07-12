from collections import namedtuple

Param = namedtuple('Param', ['DB', 'acc_method', 'DA', 'liq_method', 'DBE'])

class Params(object):
    ''' Parameter object
    '''
    def __init__(self, DB, acc_method, DA, liq_method, DBE, props={}, props_name=''):
        self.__set_params(DB, acc_method, DA, liq_method, DB)
        self.__set_props(props)
        self.__set_props_name(props_name)

    def __get_params(self):
        return self.__params
        
    def __set_params(self, DB, acc_method, DA, liq_method, DBE):
        self.__params = Param(DB, acc_method, DA, liq_method, DB)

    def __get_props(self):
        return self.__props

    def __set_props(self, props):
        self.__props = props

    def __get_props_name(self):
        return self.__props_name

    def __set_props_name(self, props_name):
        self.__props_name = props_name
        
    params     = property(__get_params, __set_params)
    props      = property(__get_props,  __set_props)
    props_name = property(__get_props_name, __set_props_name)
        
class Problem_1(Params):
    def __init__(self):
        super(Problem_1, self).__init__(5, 'linear', 5, 'linear', 2)

class Problem_2(Params):
    def __init__(self):
        super(Problem_2, self).__init__(10, 'linear', 10, 'linear', 2)        

class Problem_3(Params):
    def __init__(self):
        super(Problem_3, self).__init__(2, 'linear', 10, 'linear', 2)

class Problem_4(Params):
    def __init__(self):
        props = dict(liq_if_est_money=True, liq_cutoff=0.0)
        props_name = 'prop_0'
        super(Problem_4, self).__init__(5, 'linear', 5, 'linear', 2, props=props, props_name=props_name)
        
class Problem_5(Params):
    def __init__(self):
        props = dict(liq_if_est_money=True, liq_cutoff=0.0)
        props_name = 'prop_0'        
        super(Problem_5, self).__init__(3, 'linear', 10, 'linear', 2, props=props, props_name=props_name)

class Problem_6(Params):
    def __init__(self):
        props = dict(liq_if_est_money=True, liq_cutoff=0.0)
        props_name = 'prop_0'                
        super(Problem_6, self).__init__(5, 'linear', 10, 'linear', 2, props=props, props_name=props_name)

class Problem_7(Params):
    def __init__(self):
        props = dict(liq_if_est_money=True, liq_cutoff=0.0)
        props_name = 'prop_0'                        
        super(Problem_7, self).__init__(10, 'linear', 10, 'linear', 2, props=props, props_name=props_name)

GSCIParamList = [Problem_1(), Problem_2(), Problem_3(), Problem_4(), Problem_5(), Problem_6(), Problem_7()]
