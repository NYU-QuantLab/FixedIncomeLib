from typing import Union, Dict, Any, Tuple
from datetime import datetime
from fixedincomelib.market import *
from fixedincomelib.date import (TermOrTerminationDate, Date, add_period)
from fixedincomelib.product.product_interfaces import ProductBuilderRegistry
from fixedincomelib.product.linear_products import (ProductRFRFuture, ProductRFRSwap)
from fixedincomelib.product.utilities import (LongOrShort, PayOrReceive)

class ProductFactory:

    @classmethod
    def createProductFromDataConvention(
        cls,
        value_date : Date,
        axis1: str,
        data_convention: DataConvention,
        values: float,
        **kwargs: Any):

        convention_obj : DataConvention = data_convention
        prod_type = convention_obj.type()
        func = ProductBuilderRegistry().get(prod_type)
        return func(value_date, axis1, convention_obj, values, **kwargs)

    @classmethod
    def create_rfr_future(
        cls,
        value_date : Date,
        axis1: str,
        data_convention: DataConventionRFRFuture,
        values: float,
        **kwargs: Any) -> ProductRFRFuture:

        term_or_effective_date, term_or_termnation_date = \
            ProductFactory._tokenize_axis1(axis1)
        if term_or_effective_date.is_term():
            raise Exception('Effective date is not valid.')
        if term_or_termnation_date is None:
            raise Exception('Term or Termination date is missing.')
        long_or_short = LongOrShort.from_string(kwargs.get('long_or_short', 'long'))
        amount = kwargs.get('amount', data_convention.contractual_notional)
        return ProductRFRFuture(
            effective_date=term_or_effective_date.get_date(),
            term_or_termination_date=term_or_termnation_date,
            future_conv=data_convention.name,
            long_or_short=long_or_short,
            amount=amount,
            strike=values)

    @classmethod
    def create_rfr_swap(
        cls,
        value_date : Date,
        axis1: str,
        data_convention: DataConventionRFRSwap,
        values: float,
        **kwargs: Any) -> ProductRFRSwap:

        term_or_effective_date, term_or_termination_date = \
              ProductFactory._tokenize_axis1(axis1)
        
        pay_offset = data_convention.payment_offset
        on_index_str = data_convention.index_str
        on_index = data_convention.index
        accrual_period = data_convention.acc_period
        accrual_basis = data_convention.acc_basis
        pay_buinsess_day_convention = data_convention.business_day_convention
        pay_hoiday_convention = data_convention.holiday_convention 
        pay_or_rec = kwargs.get('pay_or_rec', 'receive')
        spread = kwargs.get('spread', 0.)
        compounding_method = CompoundingMethod.from_string(kwargs.get('compound_method', 'compound'))

        effective_date = value_date
        if term_or_termination_date is None:
            # spot starting
            effective_date = on_index.fixingDate(value_date)
            term_or_termination_date = term_or_effective_date
        else:
            # forwad starting
            if term_or_effective_date.is_term():
                this_date = on_index.fixingDate(value_date)
                effective_date = add_period(
                    this_date,
                    term_or_effective_date.get_term(),
                    on_index.businessDayConvention(),
                    pay_hoiday_convention) # fix holiday convention
            else:
                effective_date = term_or_effective_date.get_date()

        return ProductRFRSwap(
            effective_date=effective_date,
            term_or_termination_date=term_or_termination_date,
            payment_off_set=pay_offset,
            on_index=on_index_str,
            fixed_rate=values,
            pay_or_rec=PayOrReceive.from_string(pay_or_rec),
            notional=kwargs.get('notinoal', 1e6),
            accrual_period=accrual_period,
            accrual_basis=accrual_basis,
            floating_leg_accrual_period=accrual_period,
            pay_business_day_convention=pay_buinsess_day_convention,
            pay_holiday_convention=pay_hoiday_convention,
            spread=spread,
            compounding_method=compounding_method)

    ### utilities    
    @staticmethod
    def _tokenize_axis1(axis1: str):
        
        axis1 = axis1.strip()
        if "x" in axis1.lower():
            tokens = axis1.replace("X", "x").split("x")
            return TermOrTerminationDate(tokens[0]), TermOrTerminationDate(tokens[1])
        else:
            return TermOrTerminationDate(axis1), None

### support product factory
ProductBuilderRegistry().register(DataConventionRFRFuture.type(), ProductFactory.create_rfr_future)
ProductBuilderRegistry().register(DataConventionRFRSwap.type(), ProductFactory.create_rfr_swap)
