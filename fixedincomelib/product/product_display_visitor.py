from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
from fixedincomelib.product.product import Product, ProductVisitor
from fixedincomelib.product.portfolio import ProductPortfolio
from fixedincomelib.product.linear_products import (
    ProductBulletCashflow, ProductIborCashflow, ProductOvernightIndexCashflow,
    ProductFuture, ProductRfrFuture, InterestRateStream,
    ProductIborSwap, ProductOvernightSwap)
from fixedincomelib.product.non_linear_products import (
    ProductIborCapFloorlet, ProductOvernightCapFloorlet,
    CapFloorStream, ProductIborCapFloor, ProductOvernightCapFloor,
    ProductIborSwaption, ProductOvernightSwaption)

# ------------------------
# Utility functions
# ------------------------
def _iso(x: Any) -> Any:
    """Convert QuantLib.Date or similar to ISO string if possible."""
    try:
        return x.ISO()
    except Exception:
        return x

def _ccy(prod: Product) -> str:
    """Extract currency code from product."""
    try:
        return prod.currency.value.code()
    except Exception:
        return str(prod.currency)

def _add(rows: List[Tuple[str, Any]], key: str, val: Any):
    """Append a row to key-value list."""
    rows.append((key, val))

# ------------------------
# Unified Visitor
# ------------------------
class ProductDisplayVisitor(ProductVisitor):
    """
    A single unified Product display visitor.
    - All product subclasses go through visit(self, prod)
    - Dispatch table maps product class -> rendering method
    - Supports nested structures like Portfolio and InterestRateStream
    """

    def __init__(self, *, preview_children: int | None = None) -> None:
        self._rows: List[Tuple[str, Any]] = []
        self._preview_children = preview_children  # preview limit for nested children

    def visit(self, prod: Product):
        """Generic visit entrypoint for all Products."""
        rows: List[Tuple[str, Any]] = []

        # Generic header fields
        _add(rows, "Type", getattr(prod, "prodType", prod.__class__.__name__))
        _add(rows, "Currency", _ccy(prod))
        _add(rows, "Notional", prod.notional)
        _add(rows, "LongOrShort", prod.longOrShort.valueStr)
        _add(rows, "FirstDate", _iso(prod.firstDate))
        _add(rows, "LastDate", _iso(prod.lastDate))

        # Dispatch specific renderer
        handler = self._dispatch().get(type(prod))
        if handler:
            rows += handler(self, prod)
        else:
            if isinstance(prod, ProductPortfolio):
                rows += self._render_portfolio(prod)
            elif isinstance(prod, InterestRateStream):
                rows += self._render_stream(prod)

        # Cache rows and also return immediate result
        self._rows.extend(rows)
        return pd.DataFrame(rows, columns=["Attribute", "Value"])

    def display(self) -> pd.DataFrame:
        """Return all collected rows as a single DataFrame."""
        return pd.DataFrame(self._rows, columns=["Attribute", "Value"])

    # ---------- Dispatch table ----------
    def _dispatch(self) -> Dict[type, Any]:
        return {
            ProductBulletCashflow:        ProductDisplayVisitor._render_bullet_cf,
            ProductIborCashflow:          ProductDisplayVisitor._render_ibor_cf,
            ProductOvernightIndexCashflow:ProductDisplayVisitor._render_ois_cf,
            ProductFuture:                ProductDisplayVisitor._render_future,
            ProductRfrFuture:             ProductDisplayVisitor._render_rfr_future,
            ProductIborSwap:              ProductDisplayVisitor._render_ibor_swap,
            ProductOvernightSwap:         ProductDisplayVisitor._render_ois_swap,
            ProductIborCapFloorlet:       ProductDisplayVisitor._render_ibor_caplet,
            ProductOvernightCapFloorlet:  ProductDisplayVisitor._render_ois_caplet,
            ProductIborCapFloor:          ProductDisplayVisitor._render_ibor_cap,
            ProductOvernightCapFloor:     ProductDisplayVisitor._render_ois_cap,
            ProductIborSwaption:          ProductDisplayVisitor._render_ibor_swpt,
            ProductOvernightSwaption:     ProductDisplayVisitor._render_ois_swpt,
            CapFloorStream:               ProductDisplayVisitor._render_capfloor_stream,
        }

    # ---------- Renderers ----------
    def _render_bullet_cf(self, p: ProductBulletCashflow) -> List[Tuple[str, Any]]:
        return [("TerminationDate", _iso(p.terminationDate))]

    def _render_ibor_cf(self, p: ProductIborCashflow) -> List[Tuple[str, Any]]:
        return [
            ("AccrualStart", _iso(p.accrualStart)),
            ("AccrualEnd", _iso(p.accrualEnd)),
            ("AccrualFactor", p.accrualFactor),
            ("Index", p.index),
            ("Spread", p.spread),
        ]

    def _render_ois_cf(self, p: ProductOvernightIndexCashflow) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("TerminationDate", _iso(p.terminationDate)),
            ("Index", p.index),
            ("Compounding", p.compounding),
            ("Spread", p.spread),
        ]

    def _render_future(self, p: ProductFuture) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("ExpirationDate", _iso(p.expirationDate)),
            ("MaturityDate", _iso(p.maturityDate)),
            ("AccrualFactor", p.accrualFactor),
            ("Index", p.index),
            ("Strike", p.strike),
        ]

    def _render_rfr_future(self, p: ProductRfrFuture) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("MaturityDate", _iso(p.maturityDate)),
            ("AccrualFactor", p.accrualFactor),
            ("Compounding", p.compounding),
            ("Index", p.index),
            ("Strike", p.strike),
        ]

    def _render_stream(self, p: InterestRateStream) -> List[Tuple[str, Any]]:
        n = getattr(p, "count", len(p.elements))
        limit = n if self._preview_children is None else min(self._preview_children, n)
        rows: List[Tuple[str, Any]] = [("NumCashflows", n)]
        for i in range(limit):
            cf = p.element(i)
            rows.append((f"Cashflow[{i}].Type", getattr(cf, "prodType", cf.__class__.__name__)))
            start = getattr(cf, "accrualStart", getattr(cf, "effectiveDate", getattr(cf, "firstDate", None)))
            end   = getattr(cf, "accrualEnd", getattr(cf, "terminationDate", getattr(cf, "lastDate", None)))
            rows.append((f"Cashflow[{i}].Start", _iso(start)))
            rows.append((f"Cashflow[{i}].End", _iso(end)))
        if limit < n:
            rows.append(("CashflowPreview", f"{limit}/{n} shown"))
        return rows

    def _render_ibor_swap(self, p: ProductIborSwap) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("MaturityDate", _iso(p.maturityDate)),
            ("FixedRate", p.fixedRate),
            ("Index", p.index),
            ("PayFixed", p.payFixed),
        ]

    def _render_ois_swap(self, p: ProductOvernightSwap) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("MaturityDate", _iso(p.maturityDate)),
            ("FixedRate", p.fixedRate),
            ("Index", p.index),
            ("PayFixed", p.payFixed),
        ]

    def _render_ibor_caplet(self, p: ProductIborCapFloorlet) -> List[Tuple[str, Any]]:
        return [
            ("AccrualStart", _iso(p.accrualStart)),
            ("AccrualEnd", _iso(p.accrualEnd)),
            ("Index", p.index),
            ("OptionType", p.optionType),
            ("Strike", p.strike),
        ]

    def _render_ois_caplet(self, p: ProductOvernightCapFloorlet) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("MaturityDate", _iso(p.maturityDate)),
            ("Index", p.index),
            ("Compounding", p.compounding),
            ("OptionType", p.optionType),
            ("Strike", p.strike),
        ]

    def _render_capfloor_stream(self, p: CapFloorStream) -> List[Tuple[str, Any]]:
        n = getattr(p, "numProducts", len(p.elements))
        rows = [("NumCaplets", n)]
        if n > 0:
            first = p.element(0)
            rows.append(("OptionType", getattr(first, "optionType", None)))
            if hasattr(first, "compounding"):
                rows.append(("Compounding", first.compounding))
        return rows

    def _render_ibor_cap(self, p: ProductIborCapFloor) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("MaturityDate", _iso(p.maturityDate)),
            ("Index", p.index),
            ("OptionType", p.optionType),
            ("NumCaplets", len(p.capStream.products)),
        ]

    def _render_ois_cap(self, p: ProductOvernightCapFloor) -> List[Tuple[str, Any]]:
        return [
            ("EffectiveDate", _iso(p.effectiveDate)),
            ("MaturityDate", _iso(p.maturityDate)),
            ("Index", p.index),
            ("OptionType", p.optionType),
            ("Compounding", p.compounding),
            ("NumCaplets", len(p.capStream.products)),
        ]

    def _render_ibor_swpt(self, p: ProductIborSwaption) -> List[Tuple[str, Any]]:
        return [
            ("ExpiryDate", _iso(p.expiryDate)),
            ("SwapStart", _iso(p.swap.effectiveDate)),
            ("SwapEnd", _iso(p.swap.maturityDate)),
            ("Index", p.swap.index),
            ("FixedRate", p.swap.fixedRate),
            ("OptionType", p.optionType),
        ]

    def _render_ois_swpt(self, p: ProductOvernightSwaption) -> List[Tuple[str, Any]]:
        return [
            ("ExpiryDate", _iso(p.expiryDate)),
            ("SwapStart", _iso(p.swap.effectiveDate)),
            ("SwapEnd", _iso(p.swap.maturityDate)),
            ("Index", p.swap.index),
            ("FixedRate", p.swap.fixedRate),
            ("OptionType", p.optionType),
        ]

    def _render_portfolio(self, p: ProductPortfolio) -> List[Tuple[str, Any]]:
        rows: List[Tuple[str, Any]] = []
        n = p.numProducts
        _add(rows, "NumProducts", n)
        limit = n if self._preview_children is None else min(self._preview_children, n)
        for i, (child, w) in enumerate(p.elements[:limit], start=1):
            _add(rows, f"Element[{i}].Type", getattr(child, "prodType", child.__class__.__name__))
            _add(rows, f"Element[{i}].Weight", w)
        if limit < n:
            _add(rows, "ElementsPreview", f"{limit}/{n} shown")
        return rows
