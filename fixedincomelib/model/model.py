from token import OP
import pandas as pd
import datetime as dt
from typing import Any, Optional, Dict
from abc import ABCMeta, abstractmethod
from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.model.build_method import *

### restrict admissible model sets
class ModelType:

    # ordered by hierachy
    YIELD_CURVE = 1
    IR_SABR = 2
    INVALID = 3

    def __init__(self, model_type : str) -> None:
        self.value_str_ = model_type
        self.value_ = ModelType.INVALID
        if model_type.upper() == 'YIELD_CURVE':
            self.value_ =  ModelType.YIELD_CURVE
        elif model_type.upper() == 'IR_SABR':
            self.value_ =  ModelType.IR_SABR
        else:
            raise Exception('Model type ' + model_type + ' is not supported.')
    
    @property
    def value(self):
        return self.value_
    
    @property
    def value_str(self):
        return self.value_str_

    @property
    def order(self):
        return int(self.value)

### one model consist of multiple components
class ModelComponent(metaclass=ABCMeta):

    def __init__(self, value_date : Date, calibration_data : DataCollection, build_method : BuildMethod) -> None:
        self.valueDate_ = value_date
        self.calibration_data_ = calibration_data
        self.build_method_ = build_method
        self.target_ = build_method.target
        self.state_vars_ = []

    @property
    def target(self):
        return self.target_
    
    @property
    def calibraion_data(self):
        return self.calibration_data_
    
    @property
    def build_method(self):
        return self.build_method_

### model interface
class Model(metaclass=ABCMeta):

    def __init__(self, value_date : Date, model_type : ModelType, sub_model : Optional[Any]=None) -> None:
        self.value_date_ = value_date
        self.model_type_ = model_type
        self.components : Dict[str, ModelComponent] = {}
        self.sub_model_ = None        
        
    @property    
    def value_date(self):
        return self.value_date_
    
    @property
    def model_type(self):
        return self.model_type_
    
    @property
    def sub_model(self):
        return self.sub_model_

    def set_model_component(self, target : str, model_component : ModelComponent):
        self.components[target] = model_component

    def retrieve_model_component(self, target : str):
        if target.upper() in self.components:
            return self.components[target.upper()]
        else:
            raise Exception(f'This model does not contain {target.upper()} component.')
