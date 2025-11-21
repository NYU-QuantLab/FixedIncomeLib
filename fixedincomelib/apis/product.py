from typing import Optional
import pandas as pd
# native
from fixedincomelib.product import *

def displayProduct(product : Product):
    this_displayer = ProductDisplayVisitor()
    product.accept(this_displayer)
    return this_displayer.display()

def createProductFromDataConvention(
    axis1: str,
    dataConvention: str,
    values: float,
    **kwargs):

    return ProductFactory.createProductFromDataConvention(
        axis1, dataConvention, values, **kwargs
    )

def createProdcutRFRFuture(
    effectiveDate: str,
    termOrEnd: str,
    index: str, # sofr-1b
    compounding: str, # compound / average
    longOrShort: str,
    strike: float=0.0, # optional
    notional: Optional[float] = None, # notional amount
    contractualSize: Optional[float] = None,
    accrued_flag: float=-1.0):

    return ProductRfrFuture(
        effectiveDate,
        termOrEnd,
        index,
        compounding,
        longOrShort,
        strike,
        notional,
        contractualSize,
        accrued_flag)

def createProductRFRSwap(
        effectiveDate: str,
        termOrEnd: str,
        index: str,
        fixedRate: float,
        position: str,
        notional: Optional[float] = None,
        ois_spread: float = 0.0,
        compounding: Optional[str] = None):

    return ProductRfrSwap(
        effectiveDate,
        termOrEnd,
        index,
        fixedRate,
        position,
        notional,
        ois_spread,
        compounding)