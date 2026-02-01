import QuantLib as ql
from click import Option
from construct import this
import numpy as np
from typing import Optional, List
from fixedincomelib.date import (Date, Period, TermOrTerminationDate)
from fixedincomelib.date.utilities import accrued
from fixedincomelib.market.data_conventions import CompoundingMethod
from fixedincomelib.market.registries import (IndexFixingsManager, IndexRegistry)
from fixedincomelib.product.utilities import LongOrShort
from fixedincomelib.valuation import *
from fixedincomelib.product import (ProductBulletCashflow, ProductRFRFuture, ProductOvernightIndexCashflow, ProductFixedAccrued, InterestRateStream, ProductRFRSwap)
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
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
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
            self.value_,
            self.df_)
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
        self.currency_ = product.currency
        self.effective_date_ = product.effective_date
        self.termination_date_ = product.termination_date
        self.strike_ = product.strike
        self.sign_ = 1. if product.long_or_short == LongOrShort.LONG else -1.
        self.notional_ = product.notional
        self.on_index_ = product.on_index

        # resolve valuation parameters
        self.vpc_ : ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_ : FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(FundingIndexParameter._vp_type)
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)

        # self.index_engine_ = None
        # if self.value_date <= self.expiry_date_:
        tortd = TermOrTerminationDate(self.termination_date_.ISO())
        self.index_engine_ = ValuationEngineAnalyticsOvernightIndex(
            self.model_,
            self.vpc_,
            self.on_index_,
            self.effective_date_,
            tortd,
            CompoundingMethod.COMPOUND
        )

        self.df_ = 1.0
        self.forward_rate_ = 0.0
    
    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):

        self.dvdfwd_ = 0.
        self.dvddf_ = 0.
        self.payoff_ = 0.

        if self.value_date <= self.effective_date_:
            self.index_engine_.calculate_value()
            self.forward_rate_ = self.index_engine_.value()
            
            self.payoff_ = self.sign_ * self.notional_ * (100. - 100. * self.forward_rate_ - self.strike_)

            if self.value_date == self.effective_date_:
                self.value_ = self.cash_ = self.payoff_
            else:
                funding_model: YieldCurve = self.model_
                self.df_ = funding_model.discount_factor(self.funding_index_, self.effective_date_)
                self.value_ = self.payoff_ * self.df_
                # risk
                self.dvdfwd_ = -100. * self.sign_ * self.notional_ * self.df_
                self.dvddf_ = self.value_ / self.payoff_

    
    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:

        local_grad = []
        self.model_.resize_gradient(local_grad)

        if self.value_date < self.effective_date_:
            self.index_engine_.calculate_risk(local_grad, scaler * self.dvdfwd_, True)
            funding_model: YieldCurve = self.model_
            funding_model.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.effective_date_,
                local_grad,
                scaler * self.dvddf_,
                True
            )

        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
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
            self.effective_date_,
            self.payoff_,
            self.value_,
            self.df_,
            fixing_date=self.termination_date_,
            start_date=self.effective_date_,
            end_date=self.termination_date_,
            index_or_fixed=self.on_index_.name(),
            index_value=self.forward_rate_)

        return this_cf
    

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_) # TODO: implement currency method for yield curve model
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report

    def par_rate_or_spread(self) -> float:
        return self.forward_rate_

class ValuationEngineInterestRateStream(ValuationEngineProduct):
    
    def __init__(self, 
                 model : YieldCurve, 
                 valuation_parameters_collection : ValuationParametersCollection, 
                 product : InterestRateStream,
                 request : ValuationRequest):
        super().__init__(model, valuation_parameters_collection, product, request)

        self.stream_: InterestRateStream = product

        self.currencies_ = product.currency
        self.currency_ = self.currencies_[0] if isinstance(self.currencies_, list) else self.currencies_

        # resolve valuation parameters
        self.vpc_ : ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_ : FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(FundingIndexParameter._vp_type)
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)

        # fixed or float
        self.fixed_rate_ = getattr(product, "fixed_rate_", None)
        self.float_index_ = getattr(product, "float_index_", None)

        # dealing with engines for floating cashflows
        self.index_engines_ : List[Optional[ValuationEngineAnalyticsOvernightIndex]] = []
        for i in range(product.num_cashflows()):
            cf = product.cashflow(i)
            if isinstance(cf, ProductOvernightIndexCashflow):
                tortd = TermOrTerminationDate(cf.termination_date_.ISO())
                self.index_engines_.append(
                    ValuationEngineAnalyticsOvernightIndex(
                        self.model_,
                        self.vpc_,
                        cf.on_index,
                        cf.effective_date,
                        tortd,
                        cf.compounding_method
                    )
                )
            else:
                self.index_engines_.append(None)
        
        # for report
        self.dfs_: List[float] = [1.0] * product.num_cashflows()
        self.payoffs_: List[float] = [0.0] * product.num_cashflows()
        self.fwds_: List[Optional[float]] = [None] * product.num_cashflows()
        self.accruals_: List[Optional[float]] = [None] * product.num_cashflows()
    
    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__
    
    def cashflow_payoff(self, cf) -> float:
        if isinstance(cf, ProductOvernightIndexCashflow):
            eng = self.index_engines_[self._cf_idx_]
            eng.calculate_value()
            fwd = eng.value()
            self.fwds_[self._cf_idx_] = fwd

            dc = cf.on_index.dayCounter()
            acc = dc.yearFraction(cf.effective_date, cf.termination_date)
            self.accruals_[self._cf_idx_] = acc

            return cf.notional * acc * (fwd + cf.spread)

        if isinstance(cf, ProductFixedAccrued):
            if self.fixed_rate_ is None:
                raise Exception("No fixed_rate in InterestRateStream")
            self.accruals_[self._cf_idx_] = cf.accrued
            return cf.notional * self.fixed_rate_ * cf.accrued
        
        raise Exception(f"Unsupported cashflow type: {type(cf)}")


    def calculate_value(self):
        self.value_ = 0.0
        self.cash_ = 0.0

        n = self.stream_.num_cashflows()
        for i in range(n):
            self._cf_idx_ = i
            cf = self.stream_.cashflow(i)

            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date

            self.dfs_[i] = 1.0
            self.payoffs_[i] = 0.0
            self.fwds_[i] = None
            self.accruals_[i] = None

            if self.value_date > pay_date: # already paid
                continue

            payoff = self.cashflow_payoff(cf)
            self.payoffs_[i] = payoff

            if self.value_date == pay_date:
                self.value_ += payoff
                self.cash_ += payoff
            else:
                df = self.model_.discount_factor(self.funding_index_, pay_date)
                self.dfs_[i] = df
                self.value_ += payoff * df

    
    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:

        local_grad = []
        self.model_.resize_gradient(local_grad)

        n = self.stream_.num_cashflows()
        for i in range(n):
            self._cf_idx_ = i
            cf = self.stream_.cashflow(i)

            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date

            if self.value_date >= pay_date:
                continue

            payoff = self.payoffs_[i]
            df = self.dfs_[i]

            # floating leg: risk from forward rate (index engine)
            if isinstance(cf, ProductOvernightIndexCashflow):
                eng = self.index_engines_[i]
                # dPV/dfwd = notional * accrual * df
                acc = self.accruals_[i]
                if acc is None:
                    dc = cf.on_index.dayCounter()
                    acc = dc.yearFraction(cf.effective_date, cf.termination_date)
                dv_dfwd = cf.notional * acc * df
                eng.calculate_risk(local_grad, scaler * dv_dfwd, True)

            # discount factor risk: PV = payoff * df -> dPV/ddf = payoff
            self.model_.discount_factor_gradient_wrt_state(
                self.funding_index_,
                pay_date,
                local_grad,
                scaler * payoff,
                True
            )

        self.model_.resize_gradient(gradient)
        if accumulate:
            for k in range(len(gradient)):
                gradient[k] += local_grad[k]
        else:
            gradient[:] = local_grad

    
    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()

        n = self.stream_.num_cashflows()
        for i in range(n):
            cf = self.stream_.cashflow(i)

            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date

            sign = 1.0 if cf.notional >= 0 else -1.0
            notional_abs = abs(cf.notional)

            fixing_date = None
            start_date = None
            end_date = None
            index_or_fixed = None
            index_value = None
            accrued_amt = self.accruals_[i]

            if isinstance(cf, ProductOvernightIndexCashflow):
                fixing_date = cf.termination_date
                start_date = cf.effective_date
                end_date = cf.termination_date
                index_or_fixed = cf.on_index.name()
                index_value = self.fwds_[i]
            elif isinstance(cf, ProductFixedAccrued):
                fixing_date = None
                start_date = cf.effective_date
                end_date = cf.termination_date
                index_or_fixed = "FIXED"
                index_value = self.fixed_rate_

            this_cf.add_row(
                0,  # leg id for standalone stream
                self.product_.product_type,
                self.val_engine_type(),
                notional_abs,
                sign,
                pay_date,
                self.payoffs_[i],
                self.payoffs_[i] * self.dfs_[i] if self.value_date < pay_date else (self.payoffs_[i] if self.value_date == pay_date else 0.0),
                self.dfs_[i],
                fixing_date=fixing_date,
                start_date=start_date,
                end_date=end_date,
                accrued=accrued_amt,
                index_or_fixed=index_or_fixed,
                index_value=index_value
            )

        return this_cf

    
    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currencies_)
        if isinstance(self.currencies_, list):
            report.set_pv(self.currency_, self.value_)
            report.set_cash(self.currency_, self.cash_)
        else:
            report.set_pv(self.currencies_, self.value_)
            report.set_cash(self.currencies_, self.cash_)
        return report

# class ValuationEngineProductRfrSwap(ValuationEngineProduct):
    
#     def __init__(self, 
#                  model : YieldCurve, 
#                  valuation_parameters_collection : ValuationParametersCollection, 
#                  product : ProductRFRSwap,
#                  request : ValuationRequest):
#         super().__init__(model, valuation_parameters_collection, product, request)


### register
ValuationEngineProductRegistry().register((YieldCurve._model_type.to_string(), 
                                           ProductBulletCashflow._product_type,
                                           AnalyticValParam._vp_type), 
                                           ValuationEngineProductBulletCashflow)

ValuationEngineProductRegistry().register((YieldCurve._model_type.to_string(), 
                                           ProductRFRFuture._product_type,
                                           AnalyticValParam._vp_type), 
                                           ValuationEngineProductRfrFuture)

ValuationEngineProductRegistry().register((YieldCurve._model_type.to_string(), 
                                          InterestRateStream._product_type,
                                          AnalyticValParam._vp_type), 
                                          ValuationEngineInterestRateStream)

# ValuationEngineProductRegistry().register((YieldCurve._model_type.to_string(), 
#                                           ProductRFRSwap._product_type,
#                                           AnalyticValParam._vp_type), 
#                                           ValuationEngineProductRFRSwap)