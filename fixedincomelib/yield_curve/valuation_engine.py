import QuantLib as ql
import numpy as np
from typing import Optional, List
from fixedincomelib.date import (Date, Period, TermOrTerminationDate)
from fixedincomelib.date.utilities import accrued
from fixedincomelib.market.data_conventions import CompoundingMethod
from fixedincomelib.market.registries import (IndexFixingsManager, IndexRegistry)
from fixedincomelib.product.utilities import LongOrShort
from fixedincomelib.valuation import *
from fixedincomelib.product import (ProductBulletCashflow, ProductRFRFuture)
from fixedincomelib.valuation.valuation_engine import ValuationRequest
from fixedincomelib.valuation.valuation_parameters import *
from fixedincomelib.yield_curve.yield_curve_model import YieldCurve
from fixedincomelib.yield_curve.valuation_engine_analytics import ValuationEngineAnalyticsOvernightIndex


class ValuationEngineProductBulletCashflow(ValuationEngineProduct):

    def __init__(self, 
                 model : YieldCurve, 
                 valuation_parameters_collection : ValuationParametersCollection, 
                 product : ProductBulletCashflow,
                 request : ValuationRequest):
        super().__init__(model, valuation_parameters_collection, product, request)
        # get info from product
        self.currency_ = product.currency
        self.termination_date_ = product.termination_date
        self.sign_ = 1. if product.long_or_short == LongOrShort.LONG else -1.
        self.notional_ = product.notional
        # resolve valuation parameters
        self.vpc_ : ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_ : FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(FundingIndexParameter._vp_type)
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)
    
    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):

        self.df_ = 1.
        if self.value_date <= self.termination_date_:
            scaler = self.sign_ * self.notional_
            if self.value_date == self.termination_date_:
                self.value_ = self.cash_ = scaler
            else:
                funding_model : YieldCurve = self.model_
                self.df_ = funding_model.discount_factor(self.funding_index_, self.termination_date_)
                self.value_ = scaler * self.df_
    
    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:

        local_grad = []
        self.model_.resize_gradient(local_grad)
        
        if self.value_date < self.termination_date_:
            funding_model : YieldCurve = self.model_
            funding_model.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.termination_date_,
                local_grad,
                scaler * self.sign_ * self.notional_,
                True)
    
        self.model_.resize_gradient(gradient)
        if accumulate:
            gradient += local_grad
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        this_cf.add_row(
            0, 
            self.product_.product_type,
            self.val_engine_type(),
            self.notional_,
            self.sign_,
            self.termination_date_,
            self.value_ / self.df_,
            self.value_)
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_) # TODO: implement currency method for yield curve model
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report


class ValuationEngineProductRfrFuture(ValuationEngineProduct):

    def __init__(self, 
                 model : YieldCurve, 
                 valuation_parameters_collection : ValuationParametersCollection, 
                 product : ProductRFRFuture,
                 request : ValuationRequest):
        super().__init__(model, valuation_parameters_collection, product, request)
        # get info from product
        self.currency_ = product.currency_
        self.on_index_ = product.on_index_
        self.effective_date_ = product.effective_date_
        self.termination_date_ = product.termination_date_
        self.strike_ = product.strike_
        self.sign_ = 1. if product.long_or_short_ == LongOrShort.LONG else -1.
        self.notional_ = product.notional_
        self.expiry_date_ = self.effective_date_

        fixing_mgr = IndexFixingsManager()
        if not fixing_mgr.exists('SOFR-1B'):
            try:
                fixing_mgr.register(
                    'SOFR-1B',
                    'fixedincomelib/fixings/sofr-1b.csv'
                )
            except Exception as e:
                pass

        # resolve valuation parameters
        self.vpc_ : ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_ : FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(FundingIndexParameter._vp_type)
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)

        # compute forward
        tortd = TermOrTerminationDate(self.termination_date_.ISO())
        self.index_engine_ = ValuationEngineAnalyticsOvernightIndex(
            self.model_,
            self.vpc_,
            self.on_index_,
            self.effective_date_,
            tortd,
            CompoundingMethod.COMPOUND
        )

        self.df_ = 1.
        self.forward_rate_ = 0.
    
    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):

        self.df_ = 1.
        self.forward_rate_ = 0.
        
        if self.value_date <= self.expiry_date_:
            self.index_engine_.calculate_value()
            self.forward_rate_ = self.index_engine_.value()
            payoff = self.sign_ * self.notional_ * (self.forward_rate_ - self.strike_)

            if self.value_date == self.expiry_date_:
                self.value_ = self.cash_ = payoff
            else:
                funding_model : YieldCurve = self.model_
                self.df_ = funding_model.discount_factor(self.funding_index_, self.expiry_date_)
                self.value_ = payoff * self.df_

    
    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:

        local_grad = []
        self.model_.resize_gradient(local_grad)
        
        if self.value_date < self.expiry_date_:
            self.index_engine_.calculate_value()
            fwd = self.index_engine_.value()

            payoff = self.sign_ * self.notional_ * (fwd - self.strike_)

            funding_model : YieldCurve = self.model_
            df = funding_model.discount_factor(self.funding_index_, self.expiry_date_)

            # risk from forward: df * d(payoff)/d(state)
            self.index_engine_.calculate_risk(
                local_grad,
                scaler * self.sign_ * self.notional_ * df,
                True
            )

            # risk from df: payoff * d(df)/d(state)
            funding_model.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.expiry_date_,
                local_grad,
                scaler * payoff,
                True
            )
    
        self.model_.resize_gradient(gradient)
        if accumulate:
            gradient += local_grad
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        this_cf.add_row(
            0, 
            self.product_.product_type,
            self.val_engine_type(),
            self.notional_,
            self.sign_,
            self.expiry_date_,
            self.value_ / self.df_,
            self.value_)
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_) # TODO: implement currency method for yield curve model
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report


### register
ValuationEngineProductRegistry().register((YieldCurve._model_type.to_string(), 
                                           ProductBulletCashflow._product_type,
                                           AnalyticValParam._vp_type), 
                                           ValuationEngineProductBulletCashflow)

### register
ValuationEngineProductRegistry().register((YieldCurve._model_type.to_string(), 
                                           ProductRFRFuture._product_type,
                                           AnalyticValParam._vp_type), 
                                           ValuationEngineProductRfrFuture)
