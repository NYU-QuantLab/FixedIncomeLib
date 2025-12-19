from typing import Union, List
from fixedincomelib.market.registries import DataConventionRegistry, IndexRegistry
from fixedincomelib.model import (BuildMethod, BuildMethodDeserializerRregistry)


class YieldCurveBuildMethod(BuildMethod):

    _version = 1
    _build_method_type = 'YIELD_CURVE'

    def __init__(self, 
                 target : str, 
                 content : Union[List, dict]):

        super().__init__(target, 'YIELD_CURVE', content)
        if self.bm_dict['INTERPOLATION METHOD'] == '':
            self.bm_dict['INTERPOLATION METHOD'] = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
        if self.bm_dict['EXTRAPOLATION METHOD'] == '':
            self.bm_dict['EXTRAPOLATION METHOD'] = 'FLAT'

    
    def get_valid_keys(self) -> set:
        return {
            'REFERENCE INDEX',
            'FIXING',
            'LIBOR FUTURE',
            'OVERNIGHT INDEX FUTURE',
            'SWAP',
            'OVERNIGHT INDEX SWAP',
            'INTERPOLATION METHOD',
            'EXTRAPOLATION METHOD'}

    @property
    def target_index(self):
        return IndexRegistry().get(self.target)

    @property
    def fixing(self):
        if self['FIXING'] == '':
            return None
        return DataConventionRegistry().get(self['FIXING'])
    
    @property
    def libor_future(self):
        if self['LIBOR FUTURE'] == '':
            return None
        return DataConventionRegistry().get(self['LIBOR FUTURE'])
    
    @property
    def overnight_index_future(self):
        if self['OVERNIGHT INDEX FUTURE'] == '':
            return None
        return DataConventionRegistry().get(self['OVERNIGHT INDEX FUTURE'])
    
    @property
    def swap(self):
        if self['SWAP'] == '':
            return None
        return DataConventionRegistry().get(self['SWAP'])
    
    @property
    def overnight_index_swap(self):
        if self['OVERNIGHT INDEX SWAP'] == '':
            return None
        return DataConventionRegistry().get(self['OVERNIGHT INDEX SWAP'])

    @property
    def interpolation_method(self):
        return self['INTERPOLATION METHOD']

    @property
    def extrapolation_method(self):
        return self['EXTRAPOLATION METHOD']


### register
BuildMethodDeserializerRregistry().register(YieldCurveBuildMethod._build_method_type, YieldCurveBuildMethod)
BuildMethodDeserializerRregistry().register(YieldCurveBuildMethod._build_method_type + '_DES', YieldCurveBuildMethod.deserialize)