from abc import ABC
import pandas as pd
from fixedincomelib.date import *
from fixedincomelib.market.registries import *
from fixedincomelib.market.basics import AccrualBasis, BusinessDayConvention, HolidayConvention

### interface
class DataConvention(ABC):

    def __init__(self, unique_name : str, type : str, content : dict):
        super().__init__()
        self.conv_name = unique_name.upper()
        self.conv_type = type.upper()
        self.content = content
        assert len(self.content) != 0
    
    @property
    def name(self):
        return self.conv_name
    
    @property
    def type(self):
        return self.conv_type

### specific examples
class DataConventionRFRFuture(DataConvention):

    type = 'RFR FUTURE'

    def __init__(self, unique_name, content):
    
        if len(content) != 7:
            raise ValueError(f"{unique_name}: content should have 7 fields, got {len(content)}")

        self.index_ = None
        self.accrual_basis_ = None
        self.accrual_period_ = None
        self.payment_offset_ = None
        self.payment_business_day_conv_ = None
        self.payment_holiday_conv_ = None
        self.contractual_notional_ = None
        
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

        super().__init__(unique_name, DataConventionRFRFuture.type, self.__dict__.copy())

    @property
    def index(self):
        return IndexRegistry().get(self.index_)
    
    @property
    def acc_basis(self):
        return AccrualBasis(self.accrual_basis_)
    
    @property
    def acc_period(self):
        return Period(self.accrual_period_)
    
    @property
    def payment_offset(self):
        return Period(self.payment_offset_)
    
    @property
    def business_day_convention(self):
        return BusinessDayConvention(self.payment_business_day_convention_)
    
    @property
    def holiday_convention(self):
        return HolidayConvention(self.payment_holiday_convention_)
    
    @property
    def contractual_notional(self):
        return self.contractual_notional_

class DataConventionRFRSwap(DataConvention):

    type = 'RFR SWAP'

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
            elif k == "PAYMENT_BUSEINSS_DAY_CONVENTION":
                self.payment_business_day_convention_ = v
            elif k == "PAYMENT_HOLIDAY_CONVENTION":
                self.payment_holiday_convention_ = v
            elif k == "COMPOUNDING_METHOD":
                self.compounding_method_ = v
                
        super().__init__(unique_name, DataConventionRFRSwap.type, self.__dict__.copy())

    @property
    def index(self):
        return IndexRegistry().get(self.index_)
    
    @property
    def acc_basis(self):
        return AccrualBasis(self.accrual_basis_)
    
    @property
    def acc_period(self):
        return Period(self.accrual_period_)
    
    @property
    def payment_offset(self):
        return Period(self.payment_offset_)
    
    @property
    def business_day_convention(self):
        return BusinessDayConvention(self.payment_business_day_convention_)
    
    @property
    def holiday_convention(self):
        return HolidayConvention(self.payment_holiday_convention_)
    
    @property
    def compounding_method(self):
        return self.compounding_method_

### registry
DataConventionRegFunction().register(DataConventionRFRFuture.type, DataConventionRFRFuture)
DataConventionRegFunction().register(DataConventionRFRSwap.type, DataConventionRFRSwap)