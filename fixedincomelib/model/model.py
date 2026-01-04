import numpy as np
import pandas as pd
import datetime as dt
import QuantLib as ql
from enum import Enum
from typing import Any, Dict
from abc import ABCMeta, abstractclassmethod
from fixedincomelib.date import *
from fixedincomelib.data import *
from fixedincomelib.product import *
from fixedincomelib.model.build_method import *

### registry for deserialization
class ModelDeserializerRegistry(Registry):
    
    def __new__(cls) -> Self:
        return super().__new__(cls, '', cls.__name__)

    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        self._map[key] = value

### registry for model builder
class ModelBuilderRegistry(Registry):
    
    def __new__(cls) -> Self:
        return super().__new__(cls, '', cls.__name__)

    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        self._map[key] = value

### restrict admissible model sets
class ModelType(Enum):
    
    YIELD_CURVE = 'YIELD_CURVE'
    IR_SABR = 'IR_SABR'

    @classmethod
    def from_string(cls, value: str) -> 'ModelType':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


### one model consist of multiple components
class ModelComponent:

    def __init__(self, 
                 value_date : Date,
                 component_identifier : ql.Index,
                 calibration_product : list[Product],
                 state_data : Any,
                 build_method : BuildMethod) -> None:
        
        self.value_date_ = value_date
        self.component_identifier_ = component_identifier
        self.calibration_product_ = calibration_product
        self.build_method_ = build_method
        self.state_data_ = state_data
        self.num_state_data_ = -1

    @property
    def value_date(self) -> Date:
        return self.value_date_

    @property
    def component_identifier(self) -> ql.IborIndex | ql.OvernightIndex:
        return self.component_identifier_

    @property
    def calibration_product(self) -> list[DataObject]:
        return self.calibration_product_
    
    @property
    def build_method(self) -> BuildMethod:
        return self.build_method_
    
    @property
    def state_data(self) -> Any:
        return self.state_data_
    
    @property
    def num_state_data(self) -> int:
        return self.num_state_data_


### model interface
class Model(metaclass=ABCMeta):

    def __init__(self, 
                 value_date : Date, 
                 model_type : ModelType,
                 data_collection : DataCollection,
                 build_method_collection : BuildMethodColleciton) -> None:
        
        self.value_date_ = value_date
        self.model_type_ = model_type
        self.data_collection_ = data_collection
        self.build_method_collection_ = build_method_collection
        self.components_ : Dict[str, ModelComponent] = {}
        self.component_indices_ : Dict[str, int] = {}
        self.sub_model_ = None        
        # risk
        self.num_components_ = 0
        self.num_sub_components_ = [] # for each component, how mnay state variables
        self.model_jacobian_ : np.ndarray = np.asarray([])

    @property    
    def value_date(self) -> Date:
        return self.value_date_
    
    @property
    def model_type(self) -> str:
        return self.model_type_.to_string()
    
    @property
    def data_collection(self) -> DataCollection:
        return self.data_collection_
    
    @property
    def build_method_collection(self) -> BuildMethodColleciton:
        return self.build_method_collection_

    @property
    def num_components(self) -> int:
        return self.num_components_
    
    @property
    def component_indices(self) -> Dict:
        return self.component_indices_

    @property
    def num_sub_components(self) -> int:
        return self.num_sub_components_

    @property
    def model_jacobian(self) -> np.ndarray:
        return self.model_jacobian_

    @property
    def sub_model(self) -> 'Model':
        return self.sub_model_

    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractclassmethod
    def deserialize(cls, input_dict : dict) -> 'Model':
        pass


    def resize_gradient(self, gradient: List[np.ndarray]):
        if len(gradient) != self.num_components:
            gradient[:] = [np.array([]) for _ in range(self.num_components)]

        for i in range(self.num_components):
            if len(gradient[i]) != self.num_sub_components[i]:
                gradient[i] = np.zeros(self.num_sub_components[i])

    def set_sub_model(self, model : 'Model') -> None:
        self.sub_model_ = model

    def set_model_component(self, target : ql.Index, model_component : ModelComponent) -> None:
        self.component_indices_[target.name()] = self.num_components_
        self.components_[target.name()] = model_component
        self.num_sub_components_.append(model_component.num_state_data)
        # increment by 1
        self.num_components_ += 1
        
    def retrieve_model_component(self, target : ql.Index) -> ModelComponent:
        if target.name() in self.components_:
            return self.components_[target.name()]
        else:
            raise Exception(f'This model does not contain {target.name()} component.')
     
