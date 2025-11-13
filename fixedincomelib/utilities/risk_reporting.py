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
    space: str = "pv",    # "pv", "quote", or "both"
):
    """
    request:
      - "value"          -> returns PV (float)
      - "firstOrderRisk" -> returns risk only
      - "all"            -> returns dict (pv + risk)

    space:
      - "pv"     -> risk in parameter space using PV-based Jacobian (default)
      - "quote"  -> risk in parameter space using quote-based Jacobian
      - "both"   -> returns both pv- and quote-space risks
    """
    ve = ValuationEngineRegistry().new_valuation_engine(model, valuation_parameters, product)

    # ---------------- PV ----------------
    if request in ("value", "all"):
        ve.calculateValue()
        _, pv = ve.value_
        pv = float(pv)
        if request == "value":
            return pv

    # ------------- FIRST ORDER RISK -------------
    if request in ("firstOrderRisk", "all"):
        ve.calculateFirstOrderRisk()
        first_order_risk = np.asarray(ve.firstOrderRisk_, dtype=float)

        space_l = space.lower()
        result = {}

        # Always include PV in "all"
        if request == "all":
            result["pv"] = pv

        if space_l == "pv":
            # J_pv is ∂(calib instrument PV) / ∂θ
            J_pv, _, _ = model.jacobian(space="pv")
            J_pv = np.asarray(J_pv, dtype=float)

            risk_param = np.linalg.solve(J_pv.T, first_order_risk)

            if request == "firstOrderRisk":
                return risk_param
            result["param_risk"] = risk_param
            return result

        elif space_l == "quote":
            # J_quote is ∂(calib instrument quote) / ∂θ
            J_quote = model.jacobian(space="quote")
            J_quote = np.asarray(J_quote, dtype=float)

            risk_quote = np.linalg.solve(J_quote.T, first_order_risk)

            if request == "firstOrderRisk":
                return risk_quote
            result["quote_risk"] = risk_quote
            return result

        elif space_l == "both":
            J_pv, _, _ = model.jacobian(space="pv")
            J_pv = np.asarray(J_pv, dtype=float)

            J_quote = model.jacobian(space="quote")
            J_quote = np.asarray(J_quote, dtype=float)

            risk_param = np.linalg.solve(J_pv.T, first_order_risk)
            risk_quote = np.linalg.solve(J_quote.T, first_order_risk)

            if request == "firstOrderRisk":
                return {"param_risk": risk_param, "quote_risk": risk_quote}

            result["param_risk"] = risk_param
            result["quote_risk"] = risk_quote
            return result

        else:
            raise ValueError("space must be one of: 'pv', 'quote', 'both'")

    raise ValueError("request must be one of: 'value', 'firstOrderRisk', 'all'")
