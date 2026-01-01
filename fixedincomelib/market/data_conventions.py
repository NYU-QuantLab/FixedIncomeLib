from abc import ABC
from enum import Enum
import pandas as pd
from fixedincomelib.date import *
from fixedincomelib.market.registries import *
from fixedincomelib.market.basics import AccrualBasis, BusinessDayConvention, HolidayConvention

class CompoundingMethod(Enum):
    
    SIMPLE = 'simple'
    ARITHMETIC = 'arithmetic'
    COMPOUND = 'compound'

    @classmethod
    def from_string(cls, value: str) -> 'CompoundingMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

### interface
class DataConvention(ABC):

    _type = ''

    def __init__(self, unique_name : str, type : str, content : dict):
        super().__init__()
        self.conv_name = unique_name.upper()
        self.conv_type = type.upper()
        self.content = content
        assert len(self.content) != 0
    
    @property
    def name(self):
        return self.conv_name
    
    @classmethod
    def type(cls):
        return cls._type
    
    def display(self):
        to_print = []
        for k, v in self.content.items():
            k_ = k
            if k_.endswith('_'):
                k_ = k[:-1]
            to_print.append([k_.upper(), v])
        return pd.DataFrame(to_print, columns=['Name', 'Value'])

### specific examples
class DataConventionRFRFuture(DataConvention):

    _type = 'RFR FUTURE'

    def __init__(self, unique_name, content):
    
        if len(content) != 9:
            raise ValueError(f"{unique_name}: content should have 9 fields, got {len(content)}")

        self.index_ = None
        self.accrual_basis_ = None
        self.accrual_period_ = None
        self.payment_offset_ = None
        self.payment_business_day_conv_ = None
        self.payment_holiday_conv_ = None
        self.compounding_method_ = None
        self.contractual_notional_ = None
        self.basis_point_ = None
        
        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == 'INDEX': 
                self.index_ = v
            elif k == 'ACCRUAL_BASIS':
                self.accrual_basis_ = v
            elif k == 'ACCRUAL_PERIOD':
                self.accrual_period_ = v
            elif k == 'PAYMENT_OFFSET':
                self.payment_offset_ = v
            elif k == 'PAYMENT_BUSINESS_DAY_CONVENTION':
                self.payment_business_day_convention_ = v
            elif k == 'PAYMENT_HOLIDAY_CONVENTION':
                self.payment_holiday_convention_ = v
            elif k == 'CONTRACTUAL_NOTIONAL':
                self.contractual_notional_ = float(v)
            elif k == 'BASIS_POINT':
                self.basis_point_ = float(v)
            elif k == 'COMPOUNDING_METHOD':
                self.compounding_method_ = v

        super().__init__(unique_name, DataConventionRFRFuture._type, self.__dict__.copy())

    @property
    def index(self) -> ql.QuantLib.OvernightIndex:
        return IndexRegistry().get(self.index_)
    
    @property
    def index_str(self) -> str:
        return self.index_

    @property
    def acc_basis(self) -> AccrualBasis:
        return AccrualBasis(self.accrual_basis_)
    
    @property
    def acc_period(self) -> Period:
        return Period(self.accrual_period_)
    
    @property
    def payment_offset(self) -> Period:
        return Period(self.payment_offset_)
    
    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention(self.payment_business_day_convention_)
    
    @property
    def holiday_convention(self) -> HolidayConvention:
        return HolidayConvention(self.payment_holiday_convention_)
    
    @property
    def contractual_notional(self) -> float:
        return self.contractual_notional_
    
    @property
    def basis_point(self) -> float:
        return self.basis_point_

    @property
    def compounding_method(self) -> CompoundingMethod:
        return self.compounding_method_

class DataConventionRFRSwap(DataConvention):

    _type = 'RFR SWAP'

    def __init__(self, unique_name, content):

        if len(content) != 7:
            raise ValueError(f"{unique_name}: content should have 7 fields, got {len(content)}")

        self.index_ = None
        self.accrual_basis_ = None
        self.accrual_period_ = None
        self.payment_offset_ = None
        self.payment_business_day_convention_ = None
        self.payment_holiday_convention_ = None
        self.ois_compounding_ = None
        
        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == "INDEX": 
                self.index_ = v
            elif k == "ACCRUAL_BASIS":
                self.accrual_basis_ = v
            elif k == "ACCRUAL_PERIOD":
                self.accrual_period_ = v
            elif k == "PAYMENT_OFFSET":
                self.payment_offset_ = v
            elif k == "PAYMENT_BUSINESS_DAY_CONVENTION":
                self.payment_business_day_convention_ = v
            elif k == "PAYMENT_HOLIDAY_CONVENTION":
                self.payment_holiday_convention_ = v
            elif k == "COMPOUNDING_METHOD":
                self.compounding_method_ = v
                
        super().__init__(unique_name, DataConventionRFRSwap._type, self.__dict__.copy())

    @property
    def index(self) -> ql.QuantLib.OvernightIndex:
        return IndexRegistry().get(self.index_)
    
    @property
    def index_str(self) -> str:
        return self.index_

    @property
    def acc_basis(self) -> AccrualBasis:
        return AccrualBasis(self.accrual_basis_)
    
    @property
    def acc_period(self) -> Period:
        return Period(self.accrual_period_)
    
    @property
    def payment_offset(self) -> Period:
        return Period(self.payment_offset_)
    
    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention(self.payment_business_day_convention_)
    
    @property
    def holiday_convention(self) -> HolidayConvention:
        return HolidayConvention(self.payment_holiday_convention_)
    
    @property
    def compounding_method(self) -> CompoundingMethod:
        return self.compounding_method_

class DataConventionRFRSwaption(DataConvention):

    _type = 'RFR SWAPTION'

    def __init__(self, unique_name, content):

        if len(content) != 4:
            raise ValueError(f"{unique_name}: content should have 4 fields, got {len(content)}")

        self.index_ = None
        self.payment_offset_ = None
        self.payment_business_day_convention_ = None
        self.payment_holiday_convention_ = None
        
        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == "INDEX": 
                self.index_ = v
            elif k == "PAYMENT_OFFSET":
                self.payment_offset_ = v
            elif k == "PAYMENT_BUSINESS_DAY_CONVENTION":
                self.payment_business_day_convention_ = v
            elif k == "PAYMENT_HOLIDAY_CONVENTION":
                self.payment_holiday_convention_ = v
                
        super().__init__(unique_name, DataConventionRFRSwaption._type, self.__dict__.copy())

    @property
    def index(self) -> ql.QuantLib.OvernightIndex:
        return IndexRegistry().get(self.index_)
    
    @property
    def index_str(self) -> str:
        return self.index_

    @property
    def payment_offset(self) -> Period:
        return Period(self.payment_offset_)
    
    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention(self.payment_business_day_convention_)
    
    @property
    def holiday_convention(self) -> HolidayConvention:
        return HolidayConvention(self.payment_holiday_convention_)

class DataConventionRFRCapFloor(DataConvention):

    _type = 'RFR CAPFLOOR'

    def __init__(self, unique_name, content):

        if len(content) != 4:
            raise ValueError(f"{unique_name}: content should have 4 fields, got {len(content)}")

        self.index_ = None
        self.payment_offset_ = None
        self.payment_business_day_convention_ = None
        self.payment_holiday_convention_ = None
        
        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == "INDEX": 
                self.index_ = v
            elif k == "PAYMENT_OFFSET":
                self.payment_offset_ = v
            elif k == "PAYMENT_BUSINESS_DAY_CONVENTION":
                self.payment_business_day_convention_ = v
            elif k == "PAYMENT_HOLIDAY_CONVENTION":
                self.payment_holiday_convention_ = v
                
        super().__init__(unique_name, DataConventionRFRSwaption._type, self.__dict__.copy())

    @property
    def index(self) -> ql.QuantLib.OvernightIndex:
        return IndexRegistry().get(self.index_)
    
    @property
    def index_str(self) -> str:
        return self.index_

    @property
    def payment_offset(self) -> Period:
        return Period(self.payment_offset_)
    
    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention(self.payment_business_day_convention_)
    
    @property
    def holiday_convention(self) -> HolidayConvention:
        return HolidayConvention(self.payment_holiday_convention_)

class DataConventionJump(DataConvention):

    _type = 'JUMP'

    def __init__(self, unique_name, content):

        if len(content) != 2:
            raise ValueError(f"{unique_name}: content should have 2 fields, got {len(content)}")

        self.index_ = None
        self.jupm_size_ = 1e4

        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == "INDEX": 
                self.index_ = v
            elif k == "JUMP_SIZE":
                self.jupm_size_ = v
                
        super().__init__(unique_name, DataConventionJump._type, self.__dict__.copy())

    @property
    def index(self) -> ql.Index:
        return IndexRegistry().get(self.index_)
    
    @property
    def jump_size(self):
        return self.jupm_size_

class DataConventionIFR(DataConvention):

    _type = 'INSTANTANEOUS FORWARD RATE'

    def __init__(self, unique_name, content):

        if len(content) != 3:
            raise ValueError(f"{unique_name}: content should have 3 fields, got {len(content)}")

        self.index_ = None
        self.jupm_size_ = 1e4

        upper_content = {k.upper(): v for k,v in content.items()}
        for k, v in upper_content.items():
            if k.upper() == "INDEX": 
                self.index_ = v
            elif k == "BUSINESS_DAY_CONVENTION":
                self.business_day_convention_ = v
            elif k == "HOLIDAY_CONVENTION":
                self.holiday_convention_ = v
                
        super().__init__(unique_name, DataConventionIFR._type, self.__dict__.copy())

    @property
    def index(self) -> ql.Index:
        return IndexRegistry().get(self.index_)
    
    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return BusinessDayConvention(self.business_day_convention_)
    
    @property
    def holiday_convention(self) -> HolidayConvention:
        return HolidayConvention(self.holiday_convention_)

### registry
DataConventionRegFunction().register(DataConventionRFRFuture._type, DataConventionRFRFuture)
DataConventionRegFunction().register(DataConventionRFRSwap._type, DataConventionRFRSwap)
DataConventionRegFunction().register(DataConventionRFRSwaption._type, DataConventionRFRSwaption)
DataConventionRegFunction().register(DataConventionRFRCapFloor._type, DataConventionRFRCapFloor)
DataConventionRegFunction().register(DataConventionJump._type, DataConventionJump)
DataConventionRegFunction().register(DataConventionIFR._type, DataConventionIFR)