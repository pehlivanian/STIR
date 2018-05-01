from collections import namedtuple

# In order:
# S = minimum Sharpe required
# R = minimum ratio up_days/down_days
# D = minimum number of days in window
# M = days before min(FND,LTD) for each series
# C = max year difference in spread contracts
# Y = minimum number of years required per study to train model

Param = namedtuple( 'Param', ['S', 'R', 'D', 'M', 'C', 'Y'])

class Problem_0(object):
    def __init__(self):
        pass

    @staticmethod
    def params():
        return Param(S=3, R=.7, D=25, M=3, C=1, Y=10)

