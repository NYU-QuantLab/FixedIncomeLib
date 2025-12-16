from typing import List, Union, Self, Any
from abc import ABC, abstractclassmethod, abstractmethod
import pandas as pd
from regex import F
from sympy import Union
from fixedincomelib.utilities.utils import Registry

class BuildMethod(ABC):

    _version = 1

    def __init__(self, target : str, build_method_type : str, content : Union[List, dict]) -> None:

        self.bm_target = target
        self.bm_type = build_method_type
        assert len(self.nvp) != 0 and len(self.nvp[0]) == 2
        self.bm_dict = content
        if isinstance(self.bm_dict, list):
            for each in content:
                self.bm_dict[each[0].upper()] = each[1]
        # validation
        valid_keys = self.get_valid_keys()
        for k, v in self.bm_dict:
            if k.upper() == 'TARGET':
                assert v != ''
            if k.upper() not in valid_keys:
                raise Exception(f'{k} is not a valid key.')
                
    @abstractmethod
    def get_valid_keys(self) -> set:
        pass
    
    def __getitem__(self, key : str):
        return self.bm_dict[key.upper()]
    
    @property
    def target(self):
        return self.bm_target
    
    @property
    def type(self):
        return self.bm_type

    @property
    def content(self):
        return self.bm_dict

    def display(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.bm_dict, orient='tight')

    def serialize(self) -> dict:
        content = {}
        content['VERSION'] = self._version
        content['TYPE'] = self.type
        valid_keys = self.get_valid_keys()
        for each in valid_keys:
            content[each.upper()] = self.bm_dict[each.upper()]
        return content

    @classmethod
    def createBuildMethodFromDict(cls, input_dict : dict):
        assert 'VERSION' in input_dict
        version = input_dict['VERSION']
        input_dict.pop('VERSION')
        assert 'TYPE' in input_dict
        type = input_dict['TYPE']
        input_dict.pop('TYPE')
        assert 'TARGET' in input_dict
        target = input_dict['TARGET']
        input_dict.pop('TARGET')
        this_dict = cls.generate_content_based_on_version(version, input_dict)
        return cls(target, type, this_dict)

    @abstractclassmethod
    def generate_content_based_on_version(cls, version : float, input_dict : dict):
        return {k.upper() : v for k, v in input_dict.items()}

class BuildMethodColleciton:

    _version = 1

    def __init__(self, bm_list : List[BuildMethod]) -> None:
        self.bm_col = {}
        for each in bm_list:
            key = f'{each.type}:{each.target}'
            self.bm_col[key] = each
        self.num_bms = len(self.bm_col)

    @property
    def num_build_methods(self):
        return self.num_bms

    def __getitem__(self, target : str, type : str) -> BuildMethod:
        key = f'{type}:{target}'
        if key not in self.bm_col:
            raise Exception(f'Cannot find {key}.')
        return self.bm_col[key]
    
    def display(self):
        content = []
        for k, _ in self.bm_col.items():
            tokenized = k.split(':')
            content.append(tokenized)
        return pd.DataFrame(content, columns=['Name', 'Value'])
    
    def serialize(self):
        content = {}
        content['VERSION'] = self._version
        content['TYPE'] = 'BUILDMETHODCOLLECTION'
        count = 0
        for _, v in self.bm_col.items():
            content[f'BUILD_MEHTOD_{count}'] = v.serialize()
            count += 1
        return content
    
    @classmethod
    def deserialize(cls, input_dict : dict):
        assert 'VERSION' in input_dict
        version = input_dict['VERSION']
        input_dict.pop('VERSION')
        assert 'TYPE' in input_dict
        type = input_dict['TYPE']
        input_dict.pop('TYPE')
        bm_list = []
        for _, v in input_dict:
            pass
            # BuildMethod.createBuildMethodFromDict()
        return cls(bm_list)

class BuildMethodRregistry(Registry):
    
    def __new__(cls) -> Self:
        return super().__new__(cls, '', cls.__name__)

    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        self._map[key] = value




# @classmethod
#     def _load_file_as_dict(cls, file_name):
#         target, version = '', -1
#         this_dict = {}
#         with open(file_name, 'r', encoding='utf-8') as f:
#             for k, v in json.load(f).items():
#                 if k.upper() == 'VERSION':
#                     version = v
#                 elif k.upper() == 'TARGET':
#                     target = v
#                 else:
#                     this_dict[k.upper()] = v     
#         return target, version, this_dict