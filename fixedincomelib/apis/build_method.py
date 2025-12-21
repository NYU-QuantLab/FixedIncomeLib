import pickle
import pandas as pd
from typing import List
from fixedincomelib.model import BuildMethod
from fixedincomelib.yield_curve import YieldCurveBuildMethod
from fixedincomelib.model import (BuildMethodColleciton, BuildMethodBuilderRregistry)

def qfCreateBuildMethod(build_method_type : str, content : dict):
    assert 'TARGET' in content
    func = BuildMethodBuilderRregistry().get(build_method_type)
    return func(content['TARGET'], content)

def qfWriteBuildMethodToFile(build_method : BuildMethod, path : str):
    this_dict = build_method.serialize()
    with open(path, 'wb') as handle:
        pickle.dump(this_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 'DONE'

def qfReadBuildMethodFromFile(path : str):
    with open(path, 'rb') as handle:
        this_dict = pickle.load(handle)
        this_key = f'{YieldCurveBuildMethod._build_method_type}_DES'
        func = BuildMethodBuilderRregistry().get(this_key)
        return func(this_dict)
         
def qfCreateModelBuildMethodCollection(bm_list : List[BuildMethod]):
    return BuildMethodColleciton(bm_list)

def qfWriteBuildMethodCollectionToFile(bmc : BuildMethodColleciton, path : str):
    this_dict = bmc.serialize()
    with open(path, 'wb') as handle:
        pickle.dump(this_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 'DONE'

def qfReadBuildMethodCollectionFromFile(path : str):
    with open(path, 'rb') as handle:
        this_dict = pickle.load(handle)
        return BuildMethodColleciton.deserialize(this_dict)