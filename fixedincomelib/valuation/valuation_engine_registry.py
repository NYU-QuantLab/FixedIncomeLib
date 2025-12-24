import QuantLib as ql
from typing import Self, Any, Optional
from enum import Enum
from fixedincomelib.date import (Date, TermOrTerminationDate)
from fixedincomelib.product import *
from fixedincomelib.market import CompoundingMethod
from fixedincomelib.model.model import Model
from fixedincomelib.utilities import Registry
from fixedincomelib.valuation.valuation_engine import (
    ValuationEngineProduct, ValuationEngineAnalytics)
from fixedincomelib.valuation.valuation_parameters import ValuationParametersCollection

class ValuationMode(Enum):

    ANALYTIC = 'ANALYTIC'
    FINITE_DIFFERENCE = 'FDM'
    MONTE_CARLO = 'MC'

    @classmethod
    def from_string(cls, value: str) -> 'ValuationMode':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class ValuationEngineProductRegistry(Registry):

    def __new__(cls) -> Self:
        return super().__new__(cls, '', cls.__name__)

    def get(self, key: Any, **args) -> Any:
        try: 
            return self._map[key]
        except:
            raise KeyError(f'no entry for key : {key}.')

    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        self._map[key] = value

    def new_valuation_engine(self,
                             model : Model,
                             product : Product,
                             valuation_parameters_collection : ValuationParametersCollection,
                             valuation_mode: ValuationMode) -> ValuationEngineProduct:
        
        key = (model.model_type, product.product_type, valuation_mode.to_string())
        engine_cls = self.get(*key)
        if engine_cls is None:
            raise KeyError(f"No engine registered for key {key}")
        
        return engine_cls(model, valuation_parameters_collection, product)

class ValuationEngineAnalyticIndexRegistry(Registry):

    def __new__(cls) -> Self:
        return super().__new__(cls, '', cls.__name__)

    def get(self, key: Any, **args) -> Any:
        try: 
            return self._map[key]
        except:
            raise KeyError(f'no entry for key : {key}.')

    def register(self, key : Any, value : Any) -> None:
        super().register(key, value)
        self._map[key] = value

    @classmethod
    def new_valuation_engine_analytic_index(
            cls,
            model: Model, 
            valuation_parameters_collection: ValuationParametersCollection,
            index : ql.Index,
            effective_date : Date,
            term_or_termination_date : TermOrTerminationDate,
            compounding_method : Optional[CompoundingMethod.COMPOUND]) -> ValuationEngineAnalytics:
        
        index_type = index.__class__.__base__.__name__
        key = (model.model_type, index_type)
        engine_cls = ValuationEngineAnalyticIndexRegistry().get(key)
        if engine_cls is None:
            raise KeyError(f"No analytic index engine registered for key {key}")
        
        return engine_cls(
            model, 
            valuation_parameters_collection, 
            index, 
            effective_date, 
            term_or_termination_date, 
            compounding_method)