from fixedincomelib.build_method import *


class YieldCurveBuildMethod(BuildMethod):


    def __init__(self, target, nvp):
        super().__init__()
        

    def future(self):
        return self.nvp['RFR FUTURE']

    ####
