from collections import namedtuple

Param = namedtuple( 'Param', ['S', 'R', 'D', 'M', 'C', 'Y'])

class Problem_0(object):
    def __init__(self):
        pass

    @staticmethod
    def params():
        return Param(S=3, R=.7, D=25, M=3, C=1, Y=10)

SeasonalParamList = [ Problem_0() ]

