import pandas as pd
from typing import Tuple, Any
from abc import ABC, abstractclassmethod, abstractmethod
from fixedincomelib.market import DataConvention

class DataObject(ABC):

    _version = 1

    def __init__(self, data_type: str, data_convention: DataConvention):
        self.data_type_ = data_type
        self.data_convention_ = data_convention
        self.data_identifier_ = (data_type, data_convention.name)

    @property
    def data_identifier(self) -> Tuple[str, str]:
        return self.data_identifier_
    
    @property
    def data_type(self) -> str:
        return self.data_type_

    @property
    def data_convention(self) -> str:
        return self.data_convention_.name

    @abstractmethod
    def display(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractclassmethod
    def deserialize(cls, input_dict : dict) -> "DataObject":
        pass
