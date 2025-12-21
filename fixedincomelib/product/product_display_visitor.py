from __future__ import annotations
from atexit import register
from typing import Any, Dict, List, Tuple
import pandas as pd
from functools import singledispatchmethod
from fixedincomelib.product.product_interfaces import Product, ProductVisitor
from fixedincomelib.product.product_portfolio import ProductPortfolio
from fixedincomelib.product.linear_products import (
    ProductBulletCashflow, 
    ProductFixedAccrued, 
    ProductOvernightIndexCashflow,
    ProductRFRFuture, 
    ProductRFRSwap)

class ProductDisplayVisitor(ProductVisitor):

    def __init__(self) -> None:
        super().__init__()
        self.nvps_ = []
    
    @singledispatchmethod
    def visit(self, product : Product):
        raise NotImplementedError(f'No visitor for {Product._product_type}')

    def display(self) -> pd.DataFrame:
        return pd.DataFrame(self.nvps_, columns=['Name', 'Value'])
    
    def _common_items(self, product : Product):
        self.nvps_ = [
            ['Product Type', product.product_type],
            ['Notional', product.notional],
            ['Currency', product.currency.value_str],
            ['Long Or Short', product.long_or_short.to_string().upper()]
        ]

    @visit.register
    def _(self, product : ProductBulletCashflow):
        self._common_items(product)
        self.nvps_.append(['Termination Date', product.termination_date.ISO()])
        self.nvps_.append(['Payment Date', product.payment_date.ISO()])
    
    @visit.register
    def _(self, product : ProductFixedAccrued):
        self._common_items(product)
        self.nvps_.append(['Effective Date', product.effective_date.ISO()])
        self.nvps_.append(['Termination Date', product.termination_date.ISO()])
        self.nvps_.append(['Accrual Basis', product.accrual_basis.value_str])
        self.nvps_.append(['Payment Date', product.payment_date.ISO()])
        self.nvps_.append(['Business Day Convention', product.business_day_convention.value_str])
        self.nvps_.append(['Holiday Convention', product.holiday_convention.value_str])

    @visit.register
    def _(self, product : ProductOvernightIndexCashflow):
        self._common_items(product)
        self.nvps_.append(['Effective Date', product.effective_date.ISO()])
        self.nvps_.append(['Termination Date', product.termination_date.ISO()])
        self.nvps_.append(['ON Index', product.on_index.name()])
        self.nvps_.append(['Compounding Method', product.compounding_method.to_string().upper()])
        self.nvps_.append(['Spread', product.spread])
        self.nvps_.append(['Payment Date', product.payment_date.ISO()])

    @visit.register
    def _(self, product : ProductRFRFuture):
        self._common_items(product)
        self.nvps_.append(['Effective Date', product.effective_date.ISO()])
        self.nvps_.append(['Termination Date', product.termination_date.ISO()])
        self.nvps_.append(['Future Convention', product.future_conv.name])
        self.nvps_.append(['Amount', product.amount])
        self.nvps_.append(['Strike', product.strike])

    @visit.register
    def _(self, product : ProductRFRSwap):
        self._common_items(product)
        self.nvps_.append(['Effective Date', product.effective_date.ISO()])
        self.nvps_.append(['Termination Date', product.termination_date.ISO()])
        self.nvps_.append(['Payment Offset', product.pay_offset.__str__()])
        self.nvps_.append(['ON Index', product.on_index.name()])
        self.nvps_.append(['Fixed Rate', product.fixed_rate])
        self.nvps_.append(['Pay Or Receive', product.pay_or_rec.to_string().upper()])
        self.nvps_.append(['Accrual Period', product.accrual_period.__str__()])
        self.nvps_.append(['Accrual Basis', product.accrual_basis.value_str])
        self.nvps_.append(['Floating Leg Accrual Period', product.floating_leg_accrual_period.__str__()])
        self.nvps_.append(['Business Day Convention', product.pay_business_day_convention.value_str])
        self.nvps_.append(['Holiday Convention', product.pay_holiday_convention.value_str])

    @visit.register
    def _(self, product : ProductPortfolio):
        self.nvps_.append(['Product Type', product.product_type])
        for i in range(product.num_elemnts):
            self.nvps_.append([f'Product {i} Type', product.element(i).product_type])
            self.nvps_.append([f'Product {i} Weight', product.weight(i)])
        