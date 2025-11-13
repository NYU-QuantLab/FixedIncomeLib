from __future__ import annotations
import numpy as np
from fixedincomelib.valuation import ValuationEngineRegistry

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def createValueReport(
    valuation_parameters,
    model,
    product,
    request: str = "all",
    space: str = "quote",
):
    ve = ValuationEngineRegistry().new_valuation_engine(model, valuation_parameters, product)

    if request in ("value", "all"):
        ve.calculateValue()
        _, pv = ve.value_
        pv = float(pv)
        if request == "value":
            return pv

    if request in ("firstOrderRisk", "all"):
        ve.calculateFirstOrderRisk()
        G = np.asarray(ve.firstOrderRisk_, dtype=float).ravel()

        J_pv, S, labels = model.jacobian(space="pv") 
        J_pv = np.asarray(J_pv, dtype=float)
        hedge_units = np.linalg.solve(J_pv.T, G)  

        J_quote = model.jacobian(space="quote")      
        J_quote = np.asarray(J_quote, dtype=float)
        bucket_risk = np.linalg.solve(J_quote.T, G)

        if request == "firstOrderRisk":
            space_l = space.lower()
            if space_l == "quote":
                return bucket_risk
            elif space_l == "pv":
                return hedge_units
            else:
                raise ValueError("space must be one of: 'pv', 'quote'")

        return {
            "pv": pv,
            "hedge_units": hedge_units,
            "bucket_risk": bucket_risk,
            "jacobian_scale": S,
            "jacobian_labels": labels,
        }

    raise ValueError("request must be one of: 'value', 'firstOrderRisk', 'all'")
