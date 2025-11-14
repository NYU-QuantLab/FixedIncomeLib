import numpy as np
from fixedincomelib.valuation import ValuationEngineRegistry

def createValueReport(valuation_parameters, model, product, request="all", space="pv"):
    ve = ValuationEngineRegistry().new_valuation_engine(model, valuation_parameters, product)

    # 1) PV
    if request in ("value", "all"):
        ve.calculateValue()
        _, pv = ve.value_
        if request == "value":
            return float(pv)

    # 2) Risk
    if request in ("firstOrderRisk", "all"):
        ve.calculateFirstOrderRisk()
        g = np.asarray(ve.firstOrderRisk_, dtype=float)  
        J_pv = np.asarray(model.jacobian(), dtype=float) 
        param_risk = np.linalg.solve(J_pv.T, g)

        if space.lower() == "pv":
            if request == "firstOrderRisk":
                return param_risk
            return {"pv": pv, "param_risk": param_risk}

        elif space.lower() == "quote":
            S = np.asarray(model.calibration_quote_sensitivity(), dtype=float)
            quote_risk = S * param_risk  

            if request == "firstOrderRisk":
                return quote_risk
            return {
                "pv": pv,
                "param_risk": param_risk,
                "quote_risk": quote_risk,  
            }

        else:
            raise ValueError("space must be 'pv' or 'quote'")

    raise ValueError("request must be one of: 'value', 'firstOrderRisk', 'all'")
