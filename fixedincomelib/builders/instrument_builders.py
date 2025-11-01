from __future__ import annotations
from typing import List
from fixedincomelib.date.classes import Date
from fixedincomelib.date.utilities import addPeriod, applyOffset
from fixedincomelib.conventions.data_conventions import (DataConventionRegistry, DataConventionRFRSwap, DataConventionRFRFuture)
from fixedincomelib.product.linear_products import ProductRfrFuture, ProductOvernightSwap
from fixedincomelib.builders.product_builder_registry import ProductBuilderRegistry

def build_rfr_future(
    conv: DataConventionRFRFuture, *,
    value_date: str,
    axis_entry,
    value: float,
    notional: float | None,
    long_or_short: str,
):
    if isinstance(axis_entry, str) and "x" in axis_entry:
        start_iso, end_token = [s.strip() for s in axis_entry.split("x", 1)]
    else:
        start_iso, end_token = axis_entry

    term_or_end: str = end_token
    use_notional = None if notional is None else float(notional)
    # return ProductRfrFuture(
    #     effectiveDate=start_iso,
    #     termOrEnd=end_iso,
    #     index=conv.index_key,
    #     compounding="COMPOUND",
    #     strike=float(value),
    #     notional=use_notional,
    #     longOrShort=long_or_short,
    # )
    return ProductRfrFuture(
        effectiveDate=start_iso,
        termOrEnd=term_or_end,
        index=conv.index_key,
        compounding="AVERAGE",
        longOrShort=long_or_short,
        strike=0.0, 
        notional=use_notional,
        contractualSize=None,
        accrued_flag=-1.0 
    )


def build_rfr_swap(
    conv: DataConventionRFRSwap, *,
    value_date: str,
    axis_entry,
    value: float,
    notional: float | None,
    long_or_short: str,
):
    spot: Date = applyOffset(value_date, conv.payment_offset, conv.payment_hol_conv, conv.payment_biz_day_conv)
    maturity: Date = addPeriod(spot, str(axis_entry).strip().upper(), conv.payment_biz_day_conv, conv.payment_hol_conv)
    use_notional = 1.0 if notional is None else float(notional)
    return ProductOvernightSwap(
        effectiveDate=spot,
        maturityDate=maturity,
        frequency=conv.accrual_period,
        overnightIndex=conv.index_key,
        spread=0.0,
        fixedRate=float(value),
        notional=use_notional,
        position=("SHORT" if long_or_short.upper() == "SHORT" else "LONG"),
        holConv=conv.payment_hol_conv,
        bizConv=conv.payment_biz_day_conv,
        accrualBasis=conv.accrual_basis,
        rule="BACKWARD",
        endOfMonth=False,
    )

_BUILDER_MAP = {
    DataConventionRFRFuture: build_rfr_future,
    DataConventionRFRSwap:   build_rfr_swap,
}
for conv_cls, fn in _BUILDER_MAP.items():
    ProductBuilderRegistry().insert(conv_cls, fn)

def create_products_from_data1d(*, value_date: str, data1d) -> List:
    if len(data1d.axis) != len(data1d.values):
        raise ValueError(f"Data1D length mismatch: axis={len(data1d.axis)} values={len(data1d.values)}")
    conv = DataConventionRegistry().get(str(data1d.data_convention))
    reg = ProductBuilderRegistry()

    prods: List = []
    for axis_entry, quote in zip(data1d.axis, data1d.values):
        prods.append(reg.new_product(
            conv,
            value_date=value_date,
            axis_entry=axis_entry,
            value=float(quote),
            notional=None,
            long_or_short="LONG",
        ))
    return prods