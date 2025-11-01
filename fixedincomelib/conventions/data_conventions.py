from __future__ import annotations
import json, os
import pandas as pd
from dataclasses import dataclass
from abc import ABC
from typing import Optional, Dict, Any, Callable
from fixedincomelib.market.basics import AccrualBasis, BusinessDayConvention, HolidayConvention
from fixedincomelib.market.indices import IndexRegistry

DATA_CONVENTION_MAP = {}

class DataConvention(ABC):
    def __init__(self, unique_name : str, data_type : str, content : dict):
        super().__init__()
        self.unique_name = unique_name.upper()
        self.data_type = data_type.upper()
        self.content = content
        assert len(self.content) != 0
    
    @property
    def name(self):
        return self.unique_name
    
    @property
    def conv_type(self):
        return self.data_type
    
    def register_data_conv(self):
        convention_map = DATA_CONVENTION_MAP.get(self.data_type)
        if convention_map is None:
            raise KeyError(f"Unknown data_type: {self.data_type}")
        return convention_map(self.unique_name, self.content)
    

class DataConventionRFRFuture(DataConvention):

    data_type = 'RFR FUTURE'

    def __init__(self, unique_name, content):
    
        if len(content) != 7:
            raise ValueError(f"{unique_name}: content should have 7 fields, got {len(content)}")

        self.index = None
        self.accrual_basis = None
        self.accrual_period = None
        self.payment_offset = None
        self.payment_biz_day_conv = None
        self.payment_hol_conv = None
        self.contractual_notional = None
        
        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == "INDEX": 
                self.index = v
            elif k == "ACCRUAL_BASIS":
                self.accrual_basis = v
            elif k == "ACCRUAL_PERIOD":
                self.accrual_period = v
            elif k == "PAYMENT_OFFSET":
                self.payment_offset = v
            elif k == "PAYMENT_BIZ_DAY_CONV":
                self.payment_biz_day_conv = v
            elif k == "PAYMENT_HOL_CONV":
                self.payment_hol_conv = v
            elif k == "CONTRACTUAL_NOTIONAL":
                self.contractual_notional = float(v)

        super().__init__(unique_name, DataConventionRFRFuture.data_type, self.__dict__.copy())

DATA_CONVENTION_MAP[DataConventionRFRFuture.data_type] = DataConventionRFRFuture

class DataConventionRFRSwap(DataConvention):

    data_type = 'RFR SWAP'

    def __init__(self, unique_name, content):
        if len(content) != 8:
            raise ValueError(f"{unique_name}: content should have 8 fields, got {len(content)}")

        self.index = None
        self.accrual_basis = None
        self.accrual_period = None
        self.payment_offset = None
        self.payment_biz_day_conv = None
        self.payment_hol_conv = None
        self.ois_compounding = None
        self.contractual_notional = None
        
        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == "INDEX": 
                self.index = v
            elif k == "ACCRUAL_BASIS":
                self.accrual_basis = v
            elif k == "ACCRUAL_PERIOD":
                self.accrual_period = v
            elif k == "PAYMENT_OFFSET":
                self.payment_offset = v
            elif k == "PAYMENT_BIZ_DAY_CONV":
                self.payment_biz_day_conv = v
            elif k == "PAYMENT_HOL_CONV":
                self.payment_hol_conv = v
            elif k == "OIS_COMPOUNDING":
                self.ois_compounding = v
            elif k == "CONTRACTUAL_NOTIONAL":
                self.contractual_notional = float(v)
                
        super().__init__(unique_name, DataConventionRFRSwap.data_type, self.__dict__.copy())

DATA_CONVENTION_MAP[DataConventionRFRSwap.data_type] = DataConventionRFRSwap


class DataConventionRegistry:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._map = {}
            here = os.path.dirname(__file__)
            default_path = os.path.join(here, "data_convention_registry.json")
            if os.path.exists(default_path):
                obj.load_json(default_path)
            cls._instance = obj
        return cls._instance
    
    def insert(self, conv: DataConvention) -> None:
        key = conv.unique_name.upper()
        if key in self._map: raise ValueError(f"duplicate unique_name '{conv.unique_name}'")
        self._map[key] = conv

    def get(self, unique_name: str) -> DataConvention:
        try: return self._map[unique_name.upper()]
        except KeyError as e: raise KeyError(f"no entry for '{unique_name}'") from e

    def erase(self, unique_name: str) -> None:
        key = unique_name.upper()
        if key not in self._map:
            raise KeyError(f"No entry for '{unique_name}'")
        self._map.pop(key)

    def load_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Dict[str, Any]] = json.load(f)
            for unique_name, payload in raw.items():
                data_type = payload['kind']
                payload.pop('kind')
                self.insert(DATA_CONVENTION_MAP[data_type](unique_name, payload))
    
    def list_all_convs(self):
        return self._map


