import pandas as pd
from typing import List
from fixedincomelib.data import *


def createData1D(data_type : str, data_conv : str, df : pd.DataFrame):
    ###
    ###
    return Data1D()
    pass

def createDataCollection(data_objects : List[MarketData]):

    return DataCollection()

def displayDataCollection(data_collection : DataCollection):
    
    # loop through data objects, and return me

    #  | Data Type | Data Convention

    pass

def getDataFromDataCollection(data_collection : DataCollection, data_type  :str, data_conv : str) -> MarketData:
    pass

#### jupyrer notebook 
#### datacollection
#### data object
#### pandas
#### createData1D = > object
####
#### DataColleciton

def createModifiedVersionOfDataCollection(base_data_collection, alternation_data_collection, removals):

    # i give you two data collections
    # you merge them uniquely
    # offer use the flexiblity of remove (dataType, dataConv)
    pass