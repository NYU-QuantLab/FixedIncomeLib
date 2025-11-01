import pandas as pd
from fixedincomelib.market.basics import AccrualBasis, BusinessDayConvention, HolidayConvention
from fixedincomelib.product.product import LongOrShort, ProductVisitor, Product
from fixedincomelib.date import (Date, Period, TermOrTerminationDate)
from fixedincomelib.market import (IndexRegistry, Currency)
from typing import List, Optional, Union
from date.utilities import makeSchedule,accrued
from product.portfolio import ProductPortfolio
from conventions.data_conventions import DataConventionRegistry

# -------------------------
# Atomic Cash-Flow Classes
# -------------------------

class ProductBulletCashflow(Product):
    prodType = "ProductBulletCashflow"

    def __init__(self, 
                 terminationDate : str, 
                 currency : str,
                 notional : float,
                 longOrShort : str,
                 paymentDate: Optional[Union[str, Date]] = None) -> None:
        super().__init__(Date(terminationDate), Date(terminationDate), notional, longOrShort, Currency(currency))
    
        self.paymentDate_ = Date(terminationDate) if paymentDate is None else (paymentDate if isinstance(paymentDate, Date) else Date(paymentDate))

    @property
    def terminationDate(self):
        return self.lastDate
    
    @property
    def paymentDate(self) -> Date:
        return self.paymentDate_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
class ProductIborCashflow(Product):
    prodType = "ProductIborCashflow"

    def __init__(self,
                 startDate: str,
                 endDate: str,
                 index: str,
                 spread: float,
                 notional: float,
                 longOrShort: str,
                 paymentDate: Optional[Union[str, Date]] = None) -> None:
        
        self.accrualStart_ = Date(startDate)
        self.accrualEnd_   = Date(endDate)
        self.indexKey_ = index 
        tokenized = index.split('-')
        assert len(tokenized) >= 1, f"invalid index format: '{index}'"
        tenor     = tokenized[-1]  # e.g. "3M"
        indexName = '-'.join(tokenized[:-1])
        self.iborIndex_ = IndexRegistry().get(indexName, tenor)
        self.spread_ = spread
        ccy_code = self.iborIndex_.currency().code()
        super().__init__(self.accrualStart_, self.accrualEnd_, notional, longOrShort, Currency(ccy_code))
        self.paymentDate_ = (self.accrualEnd_ if paymentDate is None else (paymentDate if isinstance(paymentDate, Date) else Date(paymentDate)))

    @property
    def index(self):
        return self.indexKey_

    @property
    def spread(self):
        return self.spread_

    @property
    def accrualStart(self):
        return self.accrualStart_

    @property
    def accrualEnd(self):
        return self.accrualEnd_
    
    @property
    def accrualFactor(self) -> float:
        return accrued(self.accrualStart_, self.accrualEnd_)
    
    @property
    def paymentDate(self) -> Date:
        return self.paymentDate_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
class ProductOvernightIndexCashflow(Product):
    prodType = "ProductOvernightIndexCashflow"

    def __init__(
        self,
        effectiveDate: str,
        termOrEnd: Union[str, TermOrTerminationDate, Date],
        index: str,
        compounding: str,
        spread: float,
        notional: float,
        longOrShort: str,
        paymentDate: Optional[Union[str, Date]] = None) -> None:
        
        self.effDate_ = Date(effectiveDate)
        self.indexKey_   = index
        self.oisIndex_  = IndexRegistry().get(index)

        if isinstance(termOrEnd, Date):
            self.endDate_ = termOrEnd
        else:
            to = (TermOrTerminationDate(termOrEnd) if isinstance(termOrEnd, TermOrTerminationDate) else TermOrTerminationDate(termOrEnd))
            self.termOrEnd_ = to

            cal = self.oisIndex_.fixingCalendar()
            if to.isTerm():
                tenor = to.getTerm()
                self.endDate_ = Date(
                    cal.advance(self.effDate_, tenor, self.oisIndex_.businessDayConvention())
                )
            else:
                self.endDate_ = to.getDate()

        self.compounding_ = compounding.upper()
        self.spread_      = spread
        ccy_code         = self.oisIndex_.currency().code()

        super().__init__(
            self.effDate_,
            self.endDate_,
            notional,
            longOrShort,
            Currency(ccy_code)
        )

        self.paymentDate_ = (self.endDate_ if paymentDate is None else (paymentDate if isinstance(paymentDate, Date) else Date(paymentDate)))

    @property
    def index(self) -> str:
        return self.indexKey_

    @property
    def compounding(self) -> str:
        return self.compounding_

    @property
    def effectiveDate(self) -> Date:
        return self.effDate_

    @property
    def terminationDate(self) -> Date:
        return self.endDate_

    @property
    def spread(self) -> float:
        return self.spread_
    
    @property
    def paymentDate(self) -> Date:
        return self.paymentDate_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)

class ProductFuture(Product):
    prodType = "ProductFuture"

    def __init__(self, 
                 effectiveDate : str,
                 index : str,
                 strike : float,
                 notional : float,
                 longOrShort : str,
                 contractualSize: Optional[float] = None) -> None:
        
        self.strike_ = strike
        self.indexKey_ = index
        self.effectiveDate_ = Date(effectiveDate)
        tokenized_index = index.split('-')
        self.tenor_ = tokenized_index[-1] # if this errors
        self.index_ = IndexRegistry()._instance.get('-'.join(tokenized_index[:-1]), self.tenor_)
        self.expirationDate_ = Date(self.index_.fixingDate(self.effectiveDate_))
        self.maturityDate_ = Date(self.index_.maturityDate(self.effectiveDate_))

         # contractual size override vs. actual accrual
        self.contractualSize_ = contractualSize
        if contractualSize is not None:
            self.accrualFactor_ = contractualSize
        else:
            self.accrualFactor_ = accrued(self.effectiveDate_, self.maturityDate_)
        
        super().__init__(self.effectiveDate_, self.maturityDate_, notional, longOrShort, Currency(self.index_.currency().code()))
     
    @property
    def expirationDate(self):
        return self.expirationDate_

    @property
    def effectiveDate(self):
        return self.effectiveDate_

    @property
    def maturityDate(self):
        return self.maturityDate_
    
    @property
    def accrualFactor(self) -> float:
        return self.accrualFactor_
    
    @property
    def strike(self):
        return self.strike_
    
    @property
    def index(self) -> str:
        # Return the original registry key string, not the QL internal name.
        return self.indexKey_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
class ProductRfrFuture(Product):
    prodType = "ProductRfrFuture"

    def __init__(self,
                 effectiveDate: str,
                 termOrEnd: Union[str, TermOrTerminationDate],
                 index: str, # sofr-1b
                 compounding: str, # compound / average
                 longOrShort: str,
                 strike: float=0.0, # optional
                 notional: Optional[float] = None, # notional amount
                 contractualSize: Optional[float] = None,
                 accrued_flag: float=-1.0
                 ) -> None:
        
        self.effDate_    = Date(effectiveDate)
        self.termOrEnd_ = termOrEnd if isinstance(termOrEnd, str) else TermOrTerminationDate(termOrEnd)
        self.indexKey_   = index
        self.strike_     = float(strike)
        self.oisIndex_   = IndexRegistry().get(index)
        self.compounding_ = compounding.upper()
        self.conv = self.getFutureConvention()
        self.notional_ = float(notional) if notional is not None else float(self.conv.contractual_notional)

        cal = self.oisIndex_.fixingCalendar()
        if self.termOrEnd_.isTerm():
            tenor = self.termOrEnd_.getTerm()
            self.maturityDate_ = Date(cal.advance(self.effDate_, tenor, self.oisIndex_.businessDayConvention()))
        else:
            self.maturityDate_ = self.termOrEnd_.getDate()
        
        if contractualSize is not None:
            self.accrualFactor_ = float(contractualSize)
        else:
            self.accrualFactor_ = self._infer_accrual_from_term(self.termOrEnd_)

        super().__init__(self.effDate_, self.maturityDate_, self.notional_, longOrShort, Currency(self.oisIndex_.currency().code()))

    @property
    def effectiveDate(self) -> Date:
        return self.effDate_

    @property
    def maturityDate(self) -> Date:
        return self.maturityDate_
    
    @property
    def accrualFactor(self) -> float:
        return self.accrualFactor_

    @property
    def strike(self) -> float:
        return self.strike_

    @property
    def index(self) -> str:
        return self.indexKey_
    
    @property
    def compounding(self) -> str:
        return self.compounding_
    
    @property
    def terminationDate(self):
        return self.lastDate

    @property
    def futureConv(self):
        return self.conv
    
    def _infer_accrual_from_term(self, termOrTerminationDate: TermOrTerminationDate, accrued_flag: float) -> float:
        if termOrTerminationDate.isTerm() and accrued_flag == -1.0:
            t = str(termOrTerminationDate.getTerm()).upper().strip()
        if t in ("1M", "1MO"): return 1.0/12.0
        if t in ("3M", "3MO"): return 0.25
        if t in ("6M", "6MO"): return 0.5
        if t in ("12M", "1Y"): return 1.0
        if t.endswith("M"):
            try:
                m = int(t[:-1])
                return m / 12.0
            except Exception:
                pass
        if t.endswith("Y"):
            try:
                y = int(t[:-1])
                return float(y)
            except Exception:
                pass

        return accrued(self.effDate_, self.maturityDate_, self.conv.day_count())

    def getFutureConvention(self):
        # based on the inputs to deduce the data convention object
        # pass
        reg = DataConventionRegistry()
        tenor = "3M"
        if isinstance(self.termOrEnd_, TermOrTerminationDate) and self.termOrEnd_.isTerm():
            tenor = self.termOrEnd_.getTerm().upper()
        key = f"SOFR-FUTURE-{tenor}"
        try:
            return reg.get(key)
        except Exception:
            return reg.get("SOFR-FUTURE-3M")

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
# --------------------------------
# Composition: Streams & Swaps
# --------------------------------

class InterestRateStream(ProductPortfolio):

    def __init__(
        self,
        startDate: str,
        endDate: str,
        frequency: str,
        iborIndex: Optional[str]       = None,
        overnightIndex: Optional[str]  = None,
        fixedRate: Optional[float]     = None,
        ois_compounding: str           = "COMPOUND",
        ois_spread: float              = 0.0,
        notional: float                = 1.0,
        position: str                  = 'LONG',
        currency: str                  = 'USD',
        holConv: str                   = 'TARGET',
        bizConv: str                   = 'MF',
        accrualBasis: str              = 'ACT/365 FIXED',
        rule: str                      = 'BACKWARD',
        endOfMonth: bool               = False
    ):

        # calendar    = HolidayConvention(holConv).value
        # bdc         = BusinessDayConvention(bizConv).value
        # dayCounter  = AccrualBasis(accrualBasis).value

        schedule = makeSchedule(startDate, endDate, frequency, holConv, bizConv, accrualBasis, rule, endOfMonth)
        prods, weights = [], []
        for row in schedule.itertuples(index=False):
            if iborIndex:
                cf = ProductIborCashflow(Date(row.StartDate), Date(row.EndDate), iborIndex, 0.0, notional, position, Date(row.PaymentDate))
            elif overnightIndex:
                cf = ProductOvernightIndexCashflow(Date(row.StartDate), Date(row.EndDate), overnightIndex, ois_compounding, ois_spread, notional, position, Date(row.PaymentDate))
            else:
                alpha_i = accrued(Date(row.StartDate), Date(row.EndDate))
                coupon_amt = notional * (fixedRate or 0.0) * alpha_i
                cf = ProductBulletCashflow(Date(row.EndDate), currency, coupon_amt, position, Date(row.PaymentDate))
            prods.append(cf)
            weights.append(1.0)

        super().__init__(prods, weights)

    def cashflow(self, i: int) -> Product:
        return self.element(i)

class ProductIborSwap(Product):
    prodType = "ProductIborSwap"

    def __init__(
        self,
        effectiveDate: str,
        maturityDate: str,
        frequency: str,
        iborIndex: str,
        spread: float,
        fixedRate: float,
        notional: float,
        position: str,
        holConv: str      = 'TARGET',
        bizConv: str      = 'MF',
        accrualBasis: str = 'ACT/365 FIXED',
        rule: str         = 'BACKWARD',
        endOfMonth: bool  = False
    ) -> None:
        
        self.iborIndexKey = iborIndex
        self.fixedRate_   = fixedRate
        self.payFixed_    = (position.upper() == 'SHORT')
        float_position = "LONG" if self.payFixed_ else "SHORT"

        self.floatingLeg = InterestRateStream(
            startDate      = effectiveDate,
            endDate        = maturityDate,
            frequency      = frequency,
            iborIndex      = iborIndex,
            overnightIndex = None,
            fixedRate      = None,
            notional       = notional,
            position       = float_position,
            holConv        = holConv,
            bizConv        = bizConv,
            accrualBasis   = accrualBasis,
            rule           = rule,
            endOfMonth     = endOfMonth
        )
        
        self.fixedLeg = InterestRateStream(
            startDate      = effectiveDate,
            endDate        = maturityDate,
            frequency      = frequency,
            iborIndex      = None,
            overnightIndex = None,
            fixedRate      = fixedRate,
            notional       = notional,
            position       = position,
            holConv        = holConv,
            bizConv        = bizConv,
            accrualBasis   = accrualBasis,
            rule           = rule,
            endOfMonth     = endOfMonth
        )

        self.notional_ = notional
        self.position_ = LongOrShort(position)
        super().__init__(
            Date(effectiveDate),
            Date(maturityDate),
            notional,
            position,
            self.floatingLeg.element(0).currency
        )

    def floatingLegCashflow(self, i: int) -> Product:
        assert 0 <= i < self.floatingLeg.count
        return self.floatingLeg.element(i)

    def fixedLegCashflow(self, i: int) -> Product:
        assert 0 <= i < self.fixedLeg.count
        return self.fixedLeg.element(i)
    
    @property
    def effectiveDate(self) -> Date:
        return self.firstDate

    @property
    def maturityDate(self) -> Date:
        return self.lastDate

    @property
    def fixedRate(self) -> float:
        return self.fixedRate_

    @property
    def index(self) -> str:
        return self.iborIndexKey

    @property
    def payFixed(self) -> bool:
        return self.payFixed_
    
    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)


class ProductOvernightSwap(Product):
    prodType = "ProductOvernightSwap"

    def __init__(
        self,
        effectiveDate: str,
        maturityDate: str,
        frequency: str,
        overnightIndex: str,
        spread: float,
        fixedRate: float,
        notional: float,
        position: str,
        holConv: str      = 'TARGET',
        bizConv: str      = 'MF',
        accrualBasis: str = 'ACT/365 FIXED',
        rule: str         = 'BACKWARD',
        endOfMonth: bool  = False
    ) -> None:
        
        self.overnightIndexKey = overnightIndex
        self.fixedRate_        = fixedRate
        self.payFixed_         = (position.upper() == 'SHORT')
        float_position = "LONG" if self.payFixed_ else "SHORT"

        self.floatingLeg = InterestRateStream(
            startDate      = effectiveDate,
            endDate        = maturityDate,
            frequency      = frequency,
            iborIndex      = None,
            overnightIndex = overnightIndex,
            fixedRate      = None,
            notional       = notional,
            position       = float_position,
            holConv        = holConv,
            bizConv        = bizConv,
            accrualBasis   = accrualBasis,
            rule           = rule,
            endOfMonth     = endOfMonth
        )
        
        self.fixedLeg = InterestRateStream(
            startDate      = effectiveDate,
            endDate        = maturityDate,
            frequency      = frequency,
            iborIndex      = None,
            overnightIndex = None,
            fixedRate      = fixedRate,
            notional       = notional,
            position       = position,
            holConv        = holConv,
            bizConv        = bizConv,
            accrualBasis   = accrualBasis,
            rule           = rule,
            endOfMonth     = endOfMonth
        )

        self.notional_ = notional
        self.position_ = LongOrShort(position)
        super().__init__(
            Date(effectiveDate),
            Date(maturityDate),
            notional,
            position,
            self.floatingLeg.element(0).currency
        )

    def floatingLegCashflow(self, i: int) -> Product:
        assert 0 <= i < self.floatingLeg.count
        return self.floatingLeg.element(i)

    def fixedLegCashflow(self, i: int) -> Product:
        assert 0 <= i < self.fixedLeg.count
        return self.fixedLeg.element(i)
    
    @property
    def effectiveDate(self) -> Date:
        return self.firstDate

    @property
    def maturityDate(self) -> Date:
        return self.lastDate

    @property
    def fixedRate(self) -> float:
        return self.fixedRate_

    @property
    def index(self) -> str:
        return self.overnightIndexKey

    @property
    def payFixed(self) -> bool:
        return self.payFixed_
    
    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)