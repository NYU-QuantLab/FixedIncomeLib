from typing import List, Dict
import pandas as pd 
from fixedincomelib.market import *

class PVCashReport:

    def __init__(self, currencies : List[Currency]) -> None:
        self.currencies_ = currencies
        self.num_currencies_ = len(currencies)
        self.pv_ = {each : 0.  for each in self.currencies}
        self.cash_ = {each : 0.  for each in self.currencies}
    
    def set_pv(self, currency : Currency, value : float):
        assert currency in self.currencies
        self.pv[currency] = value

    def set_cash(self, currency : Currency, value : float):
        assert currency in self.currencies
        self.cash[currency] = value

    def display(self) -> pd.DataFrame:
        content = []
        for currency in self.num_currencies:
            this_pv = self.pv[currency]
            this_cash = self.cash[currency]
            content.append(
                [
                    [currency.code(), 'PV', this_pv],
                    [currency.code(), 'CASH', this_cash]
                ]
            )
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