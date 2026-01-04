from typing import List, Dict, Optional
import pandas as pd 
from fixedincomelib.market import *

### PV AND CASH REPORT

class PVCashReport:

    def __init__(self, currencies : Currency|List[Currency]) -> None:
        self.currencies_ = [currencies] if not isinstance(currencies, List) else currencies
        self.num_currencies_ = len(self.currencies_)
        self.pv_ = {each : 0.  for each in self.currencies_}
        self.cash_ = {each : 0.  for each in self.currencies_}
    
    def set_pv(self, currency : Currency, value : float):
        assert currency in self.currencies
        self.pv[currency] = value

    def set_cash(self, currency : Currency, value : float):
        assert currency in self.currencies
        self.cash[currency] = value

    def display(self) -> pd.DataFrame:
        content = []
        for currency in self.currencies_:
            this_pv = self.pv[currency]
            this_cash = self.cash[currency]
            content += \
                [
                    [currency.ccy.code(), 'PV', this_pv - this_cash],
                    [currency.ccy.code(), 'CASH', this_cash]
                ]
            
        return pd.DataFrame(content, columns=['Currency', 'Type', 'Value'])

    @property
    def currencies(self) -> List[Currency]:
        return self.currencies_
    
    @property
    def num_currencies(self) -> int:
        return self.num_currencies_
    
    @property
    def pv(self) -> Dict:
        return self.pv_
    
    @property
    def cash(self) -> Dict:
        return self.cash_
    

### CASHFLOWS REPORT

class CFReportColumns(Enum):
    
    PRODUCT_TYPE = 'PRODUCT_TYPE'
    VALUATION_ENGINE_TYPE = 'VALUATION_ENGINE_TYPE'
    LEG_ID = 'LEG_ID'
    CASHFLOW_ID = 'CASHFLOW_ID'
    FIXING_DATE = 'FIXING_DATE'
    START_DATE = 'START_DATE'
    END_DATE = 'END_DATE'
    ACCRUED = 'ACCRUED'
    PAY_DATE = 'PAY_DATE'
    INDEX_OR_FIXED = 'INDEX_OR_FIXED'
    INDEX_VALUE = 'INDEX_VALUE'
    NOTIONAL = 'NOTIONAL'
    PAY_OR_RECEIVE = 'PAY_OR_RECEIVE' # pay : -1 , receive : 1
    FORECASTED_AMOUNT = 'FORECASTED_AMOUNT'
    PV = 'PV'

    @classmethod
    def from_string(cls, value: str) -> 'CFReportColumns':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class CashflowsReport:

    def __init__(self) -> None:
        self.leg_id_tracker_ = {}
        self.content_ = []
        self.schema_ = [ \
            CFReportColumns.PRODUCT_TYPE.to_string(),
            CFReportColumns.VALUATION_ENGINE_TYPE.to_string(),
            CFReportColumns.LEG_ID.to_string(),
            CFReportColumns.CASHFLOW_ID.to_string(),
            CFReportColumns.PAY_OR_RECEIVE.to_string(),
            CFReportColumns.NOTIONAL.to_string(),
            CFReportColumns.PAY_DATE.to_string(),
            CFReportColumns.FORECASTED_AMOUNT.to_string(),
            CFReportColumns.PV.to_string()]
        
    def add_row(self,
                leg_id : int,
                prod_type : str,
                val_engine_type : str,
                notional : float,
                pay_or_rec : float,
                pay_date : Date,
                forecasted_amount : float,
                pv : float,
                fixing_date : Optional[Date]=None,
                start_date : Optional[Date]=None,
                end_date :  Optional[Date]=None,
                accrued : Optional[float]=None,
                index_or_fixed : Optional[float|str]=None,
                index_value : Optional[float]=None) -> None:


        this_row = []
        
        # sort out index
        cur_cf_id = self.leg_id_tracker_.setdefault(leg_id, 0)

        # mandatory field
        this_row = [
            prod_type,
            val_engine_type,
            leg_id,
            cur_cf_id,
            pay_or_rec,
            notional,
            pay_date,
            forecasted_amount,
            pv]

        # process optional field
        if len(self.content_) == 0:
            if fixing_date is not None:
                self.schema_.append(CFReportColumns.FIXING_DATE.to_string())
                this_row.append(fixing_date)
            if start_date is not None:
                self.schema_.append(CFReportColumns.START_DATE.to_string())
                this_row.append(start_date)
            if end_date is not None:
                self.schema_.append(CFReportColumns.END_DATE.to_string())
                this_row.append(end_date)
            if accrued is not None:
                self.schema_.append(CFReportColumns.ACCRUED.to_string())
                this_row.append(accrued)
            if index_or_fixed is not None:
                self.schema_.append(CFReportColumns.INDEX_OR_FIXED.to_string())
                this_row.append(index_or_fixed)
            if index_value is not None:
                self.schema_.append(CFReportColumns.INDEX_VALUE.to_string())
                this_row.append(index_value)
        
        # consistency validation
        if len(self.content_) != 0:
            assert len(self.content_) == len(this_row)

        # finalized
        self.content_.append(this_row)
        self.leg_id_tracker_[leg_id] += 1

    def display(self):
        return pd.DataFrame(self.content_, columns=self.schema_)
    
    @property
    def content(self) -> List:
        return self.content_

    @property
    def schema(self) -> List:
        return self.schema_