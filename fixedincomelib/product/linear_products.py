from re import S
import pandas as pd
from typing import List, Optional, Union
import QuantLib as ql
from fixedincomelib.market.basics import *
from fixedincomelib.market.registries import (IndexRegistry, DataConventionRegistry)
from fixedincomelib.market.data_conventions import (CompoundingMethod, DataConventionRFRFuture)
from fixedincomelib.market import (
    Currency, AccrualBasis, BusinessDayConvention, HolidayConvention, 
    DataConventionRegistry, IndexRegistry,
    DataConventionRFRFuture
)
from fixedincomelib.product.utilities import (LongOrShort, PayOrReceive)
from fixedincomelib.product.product_interfaces import (Product, ProductVisitor, ProductBuilderRegistry)
from fixedincomelib.date import (Date, Period, TermOrTerminationDate, make_schedule, accrued)
from fixedincomelib.product.product_portfolio import ProductPortfolio

class ProductBulletCashflow(Product):
    
    _version = 1
    _product_type = 'PRODUCT_BULLET_CASHFLOW'

    def __init__(self, 
                 termination_date : Date, 
                 currency : Currency,
                 notional : float,
                 long_or_short : LongOrShort,
                 payment_date: Optional[Date]=None) -> None:
        
        super().__init__()
        self.first_date_ = self.last_date_ = termination_date
        self.long_or_short_ = long_or_short
        self.notional_ = notional
        self.currency_ = currency
        self.paymnet_date_ = self.last_date_ if payment_date is None else payment_date

    @property
    def termination_date(self) -> Date:
        return self.last_date
    
    @property
    def payment_date(self) -> Date:
        return self.paymnet_date_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
    def serialize(self) -> dict:
        content = {}
        content['VERSION'] = self._version
        content['TYPE'] = self._product_type
        content['TERMINATION_DATE'] = self.termination_date.ISO()
        content['PAYMENT_DATE'] = self.payment_date.ISO()
        content['LONG_OR_SHORT'] = self.long_or_short.to_string().upper()
        content['NOTIONAL'] = self.notional
        content['CURRENCY'] = self.currency.value_str
        return content

    @classmethod
    def deserialize(cls, input_dict) -> 'ProductBulletCashflow':
        termination_date = Date(input_dict['TERMINATION_DATE'])
        payment_date = Date(input_dict['PAYMENT_DATE'])
        long_or_short = LongOrShort.from_string(input_dict['LONG_OR_SHORT'])
        notional = float(input_dict['NOTIONAL'])
        currency = Currency(input_dict['CURRENCY'])
        return cls(termination_date, currency, notional, long_or_short, payment_date)

class ProductFixedAccrued(Product):
    
    _version = 1
    _product_type = 'PRODUCT_FIXED_ACCRUED'

    def __init__(self, 
                 effective_date : Date,
                 termination_date : Date, 
                 currency : Currency,
                 notional : float,
                 accrual_basis : AccrualBasis,
                 payment_date: Optional[Date]=None,
                 business_day_convention : Optional[BusinessDayConvention]=BusinessDayConvention('F'),
                 holiday_convention : Optional[HolidayConvention]=HolidayConvention('USGS')) -> None:
        
        super().__init__()
        self.effective_date_ = self.first_date_ = effective_date
        self.termination_date_ = self.last_date_ = termination_date
        self.long_or_short_ = LongOrShort.LONG if notional >= 0 else LongOrShort.SHORT
        self.notional_ = notional
        self.currency_ = currency
        self.accrual_basis_ = accrual_basis
        self.business_day_convention_ = business_day_convention
        self.holiday_convention_ = holiday_convention 
        self.paymnet_date_ = self.termination_date_
        if payment_date is not None:
            self.paymnet_date_ = payment_date
        # calc accrued
        self.accrued_ = accrued(
            self.effective_date_, 
            self.termination_date_, 
            self.accrual_basis_, 
            self.business_day_convention_,
            self.holiday_convention_)

    @property
    def effective_date(self) -> Date:
        return self.effective_date_

    @property
    def termination_date(self) -> Date:
        return self.termination_date_
    
    @property
    def accrual_basis(self) -> AccrualBasis:
        return self.accrual_basis_

    @property
    def payment_date(self) -> Date:
        return self.paymnet_date_

    @property
    def business_day_convention(self) -> BusinessDayConvention:
        return self.business_day_convention_
    
    @property
    def holiday_convention(self) -> HolidayConvention:
        return self.holiday_convention_
    
    @property
    def accrued(self) -> float:
        return self.accrued_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
    def serialize(self) -> dict:
        content = {}
        content['VERSION'] = self._version
        content['TYPE'] = self._product_type
        content['EFFECTIVE_DATE'] = self.effective_date.ISO()
        content['TERMINATION_DATE'] = self.termination_date.ISO()
        content['PAYMENT_DATE'] = self.payment_date.ISO()
        content['ACCRUAL_BASIS'] = self.accrual_basis.value_str
        content['BUSINESS_DAY_CONVENTION'] = self.business_day_convention.value_str
        content['HOLIDAY_CONVENTION'] = self.holiday_convention_.value_str
        content['NOTIONAL'] = self.notional
        content['CURRENCY'] = self.currency.ccy.code()
        return content

    @classmethod
    def deserialize(cls, input_dict) -> 'ProductFixedAccrued':
        effective_date = Date(input_dict['EFFECTIVE_DATE'])
        termination_date = Date(input_dict['TERMINATION_DATE'])
        payment_date = Date(input_dict['PAYMENT_DATE'])
        accrual_basis = AccrualBasis(input_dict['ACCRUAL_BASIS'])
        business_day_convention = BusinessDayConvention(input_dict['BUSINESS_DAY_CONVENTION'])
        holiday_convention = HolidayConvention(input_dict['HOLIDAY_CONVENTION'])
        notional = float(input_dict['NOTIONAL'])
        currency = Currency(input_dict['CURRENCY'])
        return cls(effective_date, termination_date, currency, notional, 
                   accrual_basis, payment_date, business_day_convention, holiday_convention)

class ProductOvernightIndexCashflow(Product):

    _version = 1
    _product_type = 'PRODUCT_OVERNIGHT_INDEX_CASHFLOW'

    def __init__(
        self,
        effective_date: Date,
        term_or_termination_date: TermOrTerminationDate,
        on_index: str,
        compounding_method: CompoundingMethod,
        spread: float,
        notional: float,
        payment_date: Optional[Date]=None) -> None:
        
        super().__init__()

        # get index
        self.on_index_str_ = on_index
        self.on_index_ : ql.QuantLib.OvernightIndex = IndexRegistry().get(self.on_index_str_)
        # sort out date
        self.first_date_ = self.effective_date_ = effective_date
        self.termination_date_ = term_or_termination_date.get_date()
        if term_or_termination_date.is_term():
            calendar : ql.QuantLib.Calendar = self.on_index_.fixingCalendar()
            self.termination_date_ = Date(calendar.advance(
                self.effective_date_, 
                term_or_termination_date.get_term(), 
                self.on_index.businessDayConvention())) # need to find a way to get biz_day_conv from index
        self.last_date_ = self.termination_date_
        self.paymentDate_ = self.termination_date_ if payment_date is None else payment_date
        # other attributes
        self.notional_ = notional
        self.long_or_short_ = LongOrShort.LONG if notional >= 0 else LongOrShort.SHORT
        self.compounding_method_ = compounding_method
        self.spread_ = spread
        self.currency_ = Currency(self.on_index_.currency().code())

    @property
    def on_index(self) -> ql.QuantLib.OvernightIndex:
        return self.on_index_

    @property
    def compounding_method(self) -> CompoundingMethod:
        return self.compounding_method_

    @property
    def effective_date(self) -> Date:
        return self.effective_date_

    @property
    def termination_date(self) -> Date:
        return self.termination_date_

    @property
    def spread(self) -> float:
        return self.spread_
    
    @property
    def payment_date(self) -> Date:
        return self.paymentDate_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)

    def serialize(self) -> dict:
        content = {}
        content['VERSION'] = self._version
        content['TYPE'] = self._product_type
        content['EFFECTIVE_DATE'] = self.effective_date.ISO()
        content['TERMINATION_DATE'] = self.termination_date.ISO()
        content['PAYMENT_DATE'] = self.payment_date.ISO()
        content['ON_INDEX'] = self.on_index_str_
        content['SPREAD'] = self.spread
        content['COMPOUNDING_METHOD'] = self.compounding_method.to_string().upper()
        content['NOTIONAL'] = self.notional
        return content

    @classmethod
    def deserialize(cls, input_dict) -> 'ProductOvernightIndexCashflow':
        effective_date = Date(input_dict['EFFECTIVE_DATE'])
        termination_date = TermOrTerminationDate(input_dict['TERMINATION_DATE'])
        payment_date = Date(input_dict['PAYMENT_DATE'])
        on_index = input_dict['ON_INDEX']
        spread = float(input_dict['SPREAD'])
        compounding_method = CompoundingMethod.from_string(input_dict['COMPOUNDING_METHOD'])
        notional = float(input_dict['NOTIONAL'])
        return cls(
            effective_date, 
            termination_date, 
            on_index, 
            compounding_method, 
            spread, 
            notional,
            payment_date)
   
class ProductRFRFuture(Product):

    _version = 1
    _product_type = 'PRODUCT_RFR_FUTURE'

    def __init__(self,
                 effective_date: Date,
                 term_or_termination_date : TermOrTerminationDate,
                 future_conv: str,
                 long_or_short: LongOrShort,
                 amount: float,
                 strike: Optional[float]=0.0) -> None:
        
        super().__init__()

        # resolve index and convention
        self.future_conv_ : DataConventionRFRFuture = \
            DataConventionRegistry().get(future_conv)
        self.on_index_ : ql.QuantLib.OvernightIndex = self.future_conv.index
        # sort out dates
        self.first_date_ = self.effective_date_ = Date(effective_date)
        self.termination_date_ = term_or_termination_date.get_date()
        if term_or_termination_date.is_term():
            calendar = self.on_index_.fixingCalendar()
            self.termination_date_ = Date(calendar.advance(
                self.effective_date_, 
                term_or_termination_date.get_term(), 
                self.on_index_.businessDayConvention()))
        self.last_date_ = self.termination_date_
        # other attributes
        self.strike_ = strike
        self.long_or_short_ = long_or_short
        self.currency_ = Currency(self.on_index_.currency().code())
        self.amount_ = amount
        self.contractual_notional_ = self.future_conv_.contractual_notional
        self.basis_point_ = self.future_conv_.basis_point
        self.notional_ = amount * self.contractual_notional_ * self.basis_point_

    @property
    def effective_date(self) -> Date:
        return self.effective_date_

    @property
    def termination_date(self) -> Date:
        return self.termination_date_
    
    @property
    def strike(self) -> float:
        return self.strike_

    @property
    def future_conv(self) -> DataConventionRFRFuture:
        return self.future_conv_
    
    @property
    def contractual_notional(self) -> float:
        return self.contractual_notional_
    
    @property
    def notional(self) -> float:
        return self.notional_
    
    @property
    def basis_point(self) -> float:
        return self.basis_point_
    
    @property
    def on_index(self) -> ql.QuantLib.OvernightIndex:
        return self.on_index_
    
    @property
    def currency(self):
        return self.currency_
    
    @property
    def long_or_short(self):
        return self.long_or_short_

    @property
    def amount(self) -> float:
        return self.amount_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)

    def serialize(self) -> dict:
        content = {}
        content['VERSION'] = self._version
        content['TYPE'] = self._product_type
        content['EFFECTIVE_DATE'] = self.effective_date.ISO()
        content['TERMINATION_DATE'] = self.termination_date.ISO()
        content['FUTURE_CONVENTION'] = self.future_conv.name
        content['LONG_OR_SHORT'] = self.long_or_short.to_string().upper()
        content['AMOUNT'] = self.amount
        content['STRIKE'] = self.strike
        return content

    @classmethod
    def deserialize(cls, input_dict) -> 'ProductRFRFuture':
        effective_date = Date(input_dict['EFFECTIVE_DATE'])
        termination_date = TermOrTerminationDate(input_dict['TERMINATION_DATE'])
        future_conv = input_dict['FUTURE_CONVENTION']
        long_or_short = LongOrShort.from_string(input_dict['LONG_OR_SHORT'])
        amount = float(input_dict['AMOUNT'])
        strike = float(input_dict['STRIKE'])
        return cls(
            effective_date, 
            termination_date,
            future_conv, 
            long_or_short, 
            amount, 
            strike)

class InterestRateStream(ProductPortfolio):

    def __init__(
        self,
        effective_date: Date,
        termination_date: Date,
        accrual_period: Period,
        notional: float,
        currency: Currency,
        accrual_basis : AccrualBasis,
        buseinss_day_convention: BusinessDayConvention,
        holiday_convention: HolidayConvention,
        float_index: Optional[str]=None,
        fixed_rate: Optional[float]=None,
        is_on_index: Optional[bool]=True,
        # has default values
        ois_compounding: Optional[CompoundingMethod]=CompoundingMethod.COMPOUND,
        ois_spread: Optional[float]=0.0,
        fixing_in_arrear : Optional[bool]=True,
        payment_offset : Optional[Period]=Period('0D'),
        payment_business_day_convention : Optional[BusinessDayConvention]=BusinessDayConvention('F'),
        payment_holiday_convention: Optional[HolidayConvention]=HolidayConvention('USGS'),
        rule: Optional[str]='BACKWARD',
        end_of_month: Optional[bool]=False):

        if float_index is None and fixed_rate is None:
            raise Exception('Cannot have both floating index and fixed rate invalid.')
        
        self.float_index_ = float_index
        self.fixed_rate_ = fixed_rate

        schedule = make_schedule(
            start_date=effective_date, 
            end_date=termination_date, 
            accrual_period=accrual_period, 
            holiday_convention=holiday_convention, 
            business_day_convention=buseinss_day_convention, 
            accrual_basis=accrual_basis,
            rule=rule, 
            end_of_month=end_of_month,
            fix_in_arrear=fixing_in_arrear, 
            payment_offset=payment_offset,
            payment_business_day_convention=payment_business_day_convention,
            payment_holiday_convention=payment_holiday_convention)
        
        products, weights = [], []
        for _, row in schedule.iterrows():
            if float_index:
                if not is_on_index:
                    # TODO : ibor
                    raise Exception('NOT IMPLEMENTED')
                else:
                    cf = ProductOvernightIndexCashflow(
                        row.StartDate,
                        TermOrTerminationDate(row.EndDate),
                        float_index, 
                        ois_compounding, 
                        ois_spread, 
                        notional,
                        row.PaymentDate)
            else:
                cf = ProductFixedAccrued(
                    row.StartDate, 
                    row.EndDate,
                    currency,
                    notional,
                    accrual_basis,
                    row.PaymentDate,
                    buseinss_day_convention,
                    holiday_convention)

            products.append(cf)
            weights.append(1.0)

        super().__init__(products, weights)

    @property
    def float_index(self) -> Optional[str]:
        return self.float_index_

    @property
    def fixed_rate(self) -> Optional[float]:
        return self.fixed_rate_

    def cashflow(self, i: int) -> Product:
        return self.element(i)
    
    def num_cashflows(self) -> int:
        return self.num_elements_
       
class ProductRFRSwap(Product):

    _version = 1
    _product_type = 'PRODUCT_RFR_SWAP'
    
    def __init__(
        self, 
        effective_date: Date,
        term_or_termination_date: TermOrTerminationDate,
        payment_off_set : Period,
        on_index: str,
        fixed_rate: float,
        pay_or_rec: PayOrReceive,
        notional: float,
        accrual_period : Period,
        accrual_basis : AccrualBasis,
        floating_leg_accrual_period : Optional[Period]=None,
        pay_business_day_convention : Optional[BusinessDayConvention]=BusinessDayConvention('F'),
        pay_holiday_convention : Optional[HolidayConvention]=HolidayConvention('USGS'),
        spread: Optional[float] = 0.0,
        compounding_method : Optional[CompoundingMethod]=CompoundingMethod.COMPOUND) -> None:

        super().__init__()

        self.on_index_str_ = on_index
        self.on_index_ : ql.QuantLib.OvernightIndex = IndexRegistry().get(self.on_index_str_)
        self.pay_business_day_convention_ = pay_business_day_convention
        self.pay_holiday_convention_ = pay_holiday_convention
        self.first_date_ = self.effective_date_ = effective_date
        self.term_or_termination_date_ = term_or_termination_date
        self.termination_date_ = self.term_or_termination_date_.get_date()
        if self.term_or_termination_date_.is_term():
            calendar = self.on_index_.fixingCalendar()
            self.termination_date_ = Date(calendar.advance(
                self.effective_date_, 
                self.term_or_termination_date_.get_term(), 
                self.on_index_.businessDayConvention()))
        self.last_date_ = self.termination_date_
        # other attributes
        self.currency_ = Currency(self.on_index_.currency().code())
        self.fixed_rate_ = fixed_rate
        self.notional_ = notional
        self.spread_ = spread
        self.pay_or_rec_ = pay_or_rec
        self.long_or_short_ = LongOrShort.LONG if notional > 0 else LongOrShort.SHORT 
        self.pay_offset_ = payment_off_set
        self.accrual_basis_ = accrual_basis
        self.accrual_period_ = accrual_period
        self.floating_leg_accrual_period_ = self.accrual_period_ if floating_leg_accrual_period is None else floating_leg_accrual_period
        self.compounding_method_ = compounding_method
        fixed_leg_sign = 1. if self.pay_or_rec_ == PayOrReceive.PAY else -1.

        # floating leg
        self.floating_leg_ = InterestRateStream(
            effective_date=self.effective_date_,
            termination_date=self.termination_date_,
            accrual_period=self.floating_leg_accrual_period_,
            notional=self.notional_ * fixed_leg_sign * -1.,
            currency=Currency(self.on_index_.currency().code()),
            accrual_basis=self.accrual_basis_,
            buseinss_day_convention=self.pay_business_day_convention_, # not the best
            holiday_convention=HolidayConvention(self.on_index_.fixingCalendar().name()),
            float_index=on_index,
            ois_compounding=self.compounding_method_,
            ois_spread=spread,
            fixing_in_arrear=True,
            payment_offset=self.pay_offset_,
            payment_business_day_convention=self.pay_business_day_convention_,
            payment_holiday_convention=self.pay_holiday_convention_)
        # fixed leg
        self.fixed_leg_ = InterestRateStream(
            effective_date=self.effective_date_,
            termination_date=self.termination_date_,
            accrual_period=self.accrual_period_,
            notional=self.notional_ * fixed_leg_sign,
            currency=Currency(self.on_index_.currency().code()),
            accrual_basis=self.accrual_basis_,
            buseinss_day_convention=self.pay_business_day_convention_,
            holiday_convention=self.pay_holiday_convention_,
            fixed_rate=self.fixed_rate_,
            is_on_index=False,
            payment_offset=self.pay_offset_,
            payment_business_day_convention=self.pay_business_day_convention_,
            payment_holiday_convention=self.pay_holiday_convention_)
    
    def floating_leg_cash_flow(self, i: int) -> Product:
        assert 0 <= i < self.floating_leg_.num_cashflows()
        return self.floating_leg_.element(i)
    
    def fixed_leg_cash_flow(self, i: int) -> Product:
        assert 0 <= i < self.fixed_leg_.num_cashflows()
        return self.fixed_leg_.element(i)
    
    @property
    def effective_date(self) -> Date:
        return self.effective_date_
    
    @property
    def termination_date(self) -> Date:
        return self.termination_date_
    
    @property
    def term_or_termination_date(self) -> Date:
        return self.term_or_termination_date_

    @property
    def pay_offset(self) -> Period:
        return self.pay_offset_

    @property
    def fixed_rate(self) -> float:
        return self.fixed_rate_
    
    @property
    def spread(self) -> float:
        return self.spread_

    @property
    def on_index(self) -> ql.QuantLib.OvernightIndex:
        return self.on_index_
    
    @property
    def pay_or_rec(self) -> PayOrReceive:
        return self.pay_or_rec_
    
    @property
    def compounding_method(self) -> CompoundingMethod:
        return self.compounding_method_
    
    @property
    def accrual_period(self) -> Period:
        return self.accrual_period_
    
    @property
    def floating_leg_accrual_period(self) -> Period:
        return self.floating_leg_accrual_period_

    @property
    def accrual_basis(self) -> AccrualBasis:
        return self.accrual_basis_

    @property
    def pay_business_day_convention(self) -> BusinessDayConvention:
        return self.pay_business_day_convention_
    
    @property
    def pay_holiday_convention(self) -> HolidayConvention:
        return self.pay_holiday_convention_
    
    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
    def serialize(self) -> dict:
        content = {}
        content['VERSION'] = self._version
        content['TYPE'] = self._product_type
        content['EFFECTIVE_DATE'] = self.effective_date.ISO()
        if self.term_or_termination_date.is_term():
            content['TERM_OR_TERMINATION_DATE'] =  self.term_or_termination_date.get_term().__str__()
        else: 
            content['TERM_OR_TERMINATION_DATE'] = self.term_or_termination_date.get_date().ISO()
        content['PAYMENT_OFFSET'] = self.pay_offset.__str__()
        content['ON_INDEX'] = self.on_index_str_
        content['FIXED_RATE'] = self.fixed_rate
        content['PAY_OR_REC'] = self.pay_or_rec.to_string().upper()
        content['NOTIONAL'] = self.notional
        content['ACCRUAL_PERIOD'] = self.accrual_period.__str__()
        content['FLOATING_LEG_ACCRUAL_PERIOD'] = self.floating_leg_accrual_period.__str__()
        content['ACCRUAL_BASIS'] = self.accrual_basis.value_str
        content['PAY_BUSINESS_DAY_CONVENTION'] = self.pay_business_day_convention.value_str
        content['PAY_HOLIDAY_CONVENTION'] = self.pay_holiday_convention_.value_str
        content['SPREAD'] = self.spread
        content['COMPOUNDING_METHOD'] = self.compounding_method.to_string().upper()
        return content

    @classmethod
    def deserialize(cls, input_dict) -> 'ProductRFRFuture':
        effective_date = Date(input_dict['EFFECTIVE_DATE'])
        term_or_termination_date = TermOrTerminationDate(input_dict['TERM_OR_TERMINATION_DATE'])
        pay_offset = Period(input_dict['PAYMENT_OFFSET'])
        on_index = input_dict['ON_INDEX']
        fixed_rate = input_dict['FIXED_RATE']
        pay_or_rec = PayOrReceive.from_string(input_dict['PAY_OR_REC'])
        notional = input_dict['NOTIONAL']
        accrual_period = Period(input_dict['ACCRUAL_PERIOD'])
        floating_leg_accrual_period = Period(input_dict['FLOATING_LEG_ACCRUAL_PERIOD'])
        accrual_basis = AccrualBasis(input_dict['ACCRUAL_BASIS'])
        pay_business_day_convention = BusinessDayConvention(input_dict['PAY_BUSINESS_DAY_CONVENTION'])
        pay_holiday_convention = HolidayConvention(input_dict['PAY_HOLIDAY_CONVENTION'])
        spread = input_dict['SPREAD']
        compounding_method = CompoundingMethod.from_string(input_dict['COMPOUNDING_METHOD'])
        return cls(
            effective_date, 
            term_or_termination_date,
            pay_offset,
            on_index,
            fixed_rate,
            pay_or_rec,
            notional,
            accrual_period,
            accrual_basis,
            floating_leg_accrual_period,
            pay_business_day_convention,
            pay_holiday_convention,
            spread,
            compounding_method)


### register
ProductBuilderRegistry().register(ProductBulletCashflow._product_type, ProductBulletCashflow)
ProductBuilderRegistry().register(ProductFixedAccrued._product_type, ProductFixedAccrued)
ProductBuilderRegistry().register(ProductOvernightIndexCashflow._product_type, ProductOvernightIndexCashflow)
ProductBuilderRegistry().register(ProductRFRFuture._product_type, ProductRFRFuture)
ProductBuilderRegistry().register(ProductRFRSwap._product_type, ProductRFRSwap)
# support de-serilization
ProductBuilderRegistry().register(f'{ProductBulletCashflow._product_type}_DES', ProductBulletCashflow.deserialize)
ProductBuilderRegistry().register(f'{ProductFixedAccrued._product_type}_DES', ProductFixedAccrued.deserialize)
ProductBuilderRegistry().register(f'{ProductOvernightIndexCashflow._product_type}_DES', ProductOvernightIndexCashflow.deserialize)
ProductBuilderRegistry().register(f'{ProductRFRFuture._product_type}_DES', ProductRFRFuture.deserialize)
ProductBuilderRegistry().register(f'{ProductRFRSwap._product_type}_DES', ProductRFRSwap.deserialize)

