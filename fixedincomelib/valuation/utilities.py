from typing import Any
from fixedincomelib.model import *
from fixedincomelib.product import *
from fixedincomelib.valuation.report import PVCashReport
from fixedincomelib.valuation.valuation_engine_registry import *
from fixedincomelib.valuation.valuation_parameters import ValuationParametersCollection


def create_value_report(
    model : Model, 
    product : Product,
    valuation_parameters_collection: ValuationParametersCollection,
    request : ValuationRequest) -> Any:

    engine = ValuationEngineProductRegistry.new_valuation_engine(
        model, 
        product,
        valuation_parameters_collection, 
        request)

    engine.calculate_value()
    if request == ValuationRequest.PV_DETAILED:
        return engine.get_value_and_cash()
    elif request == ValuationRequest.FIRST_ORDER_RISK:
        return risk_calculation(engine)
    elif request == ValuationRequest.CASHFLOWS_REPORT:
        return engine.create_cash_flows_report()
    else:
        raise Exception(f'Request is not currently supported.')


def risk_calculation(engine : ValuationEngineProduct):
    gradient = []
    engine.calculate_first_order_risk(gradient, 1., False)
    # TODO: chain model jacobian
    return gradient # <- this will change

