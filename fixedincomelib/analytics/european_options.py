import math
from enum import Enum
from typing import Optional, Dict
from click import Option
from scipy.stats import norm


# from pysabr import *

class CallOrPut(Enum):
    
    CALL = 'call'
    PUT = 'put'
    INVALID = 'invalid'

    @classmethod
    def from_string(cls, value: str) -> 'CallOrPut':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class SimpleValuationMetrics:
    
    PV = 'pv'
    DELTA = 'delta'
    GAMMA = 'gamma'
    VEGA = 'vega'
    THETA = 'theta'
    STRIKE_RISK = 'strike_risk'
    IMPLIED_NORMAL_VOL = 'implied_normal_vol'
    IMPLIED_LOGNORMAL_VOL = 'implied_lognormal_vol'
    D_N_VOL_D_LN_VOL = 'd_n_vol_d_ln_vol'
    D_N_VOL_D_FORWARD = 'd_n_vol_d_forward'
    D_N_VOL_D_TTE = 'd_n_vol_d_tte'
    D_N_VOL_D_STRIKE = 'd_n_vol_d_strike'
    D_LN_VOL_D_N_VOL = 'd_ln_vol_d_n_vol'
    D_LN_VOL_D_FORWARD = 'd_ln_vol_d_forward'
    D_LN_VOL_D_TTE = 'd_ln_vol_d_tte'
    D_LN_VOL_D_STRIKE = 'd_ln_vol_d_strike'

    @classmethod
    def from_string(cls, value: str) -> 'SimpleValuationMetrics':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class EuropeanOptionAnalytics:

    @staticmethod
    def european_option_log_normal(
        forward : float,
        strike : float,
        time_to_expiry : float,
        log_normal_sigma : float,
        option_type : Optional[CallOrPut]=CallOrPut.CALL,
        calc_risk : Optional[bool]=False) -> Dict:

        res = {}

        if time_to_expiry <= 0 or log_normal_sigma <= 0:
            raise ValueError("Time to expiry and implied log-normal sigma must be positive")

        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(forward / strike) + 0.5 * log_normal_sigma**2 * time_to_expiry) / (log_normal_sigma * sqrt_t)
        d2 = d1 - log_normal_sigma * sqrt_t

        # pricing
        if option_type == CallOrPut.CALL:
            res[SimpleValuationMetrics.PV] = forward * norm.cdf(d1) - strike * norm.cdf(d2)
        elif option_type == CallOrPut.PUT:
            res[SimpleValuationMetrics.PV] = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # risk
        if calc_risk:
            res[SimpleValuationMetrics.DELTA] = norm.cdf(d1) if option_type == CallOrPut.CALL else norm.cdf(d1) - 1
            res[SimpleValuationMetrics.GAMMA] = norm.pdf(d1) / (forward * log_normal_sigma * sqrt_t)
            res[SimpleValuationMetrics.VEGA] = forward * norm.pdf(d1) * sqrt_t
            res[SimpleValuationMetrics.THETA] = -(forward * norm.pdf(d1) * log_normal_sigma) / (2 * sqrt_t)
            res[SimpleValuationMetrics.STRIKE_RISK] = -norm.cdf(d2) if option_type == CallOrPut.CALL else norm.cdf(-d2)

        return res

    @staticmethod
    def european_option_normal(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        option_type: Optional[CallOrPut]=CallOrPut.CALL,
        calc_risk: Optional[bool]=False) -> Dict:

        res = {}

        if time_to_expiry <= 0 or normal_sigma <= 0:
            raise ValueError("Time to expiry and implied normal sigma must be positive")

        sqrt_t = math.sqrt(time_to_expiry)
        d = (forward - strike) / (normal_sigma * sqrt_t)

        # pricing
        if option_type == CallOrPut.CALL:
            res[SimpleValuationMetrics.PV] = (forward - strike) * norm.cdf(d) + normal_sigma * sqrt_t * norm.pdf(d)
        elif option_type == CallOrPut.PUT:
            res[SimpleValuationMetrics.PV] = (strike - forward) * norm.cdf(-d) + normal_sigma * sqrt_t * norm.pdf(d)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # risk
        if calc_risk:
            res[SimpleValuationMetrics.DELTA] = norm.cdf(d) if option_type == CallOrPut.CALL else norm.cdf(d) - 1
            res[SimpleValuationMetrics.GAMMA] = norm.pdf(d) / (normal_sigma * sqrt_t)
            res[SimpleValuationMetrics.VEGA] = sqrt_t * norm.pdf(d)
            res[SimpleValuationMetrics.THETA] = -0.5 * normal_sigma * norm.pdf(d) / sqrt_t
            res[SimpleValuationMetrics.STRIKE_RISK] = -norm.cdf(d) if option_type == CallOrPut.CALL else norm.cdf(-d)

        return res

    @staticmethod
    def implied_lognormal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut]=CallOrPut.CALL,
        calc_risk : Optional[bool]=False,
        tol: Optional[float] = 1e-8
    ) -> Dict:

        res = {}

        # 1) compute implied vol
        sigma_imp = EuropeanOptionAnalytics._implied_lognormal_vol_black(
            pv, forward, strike, time_to_expiry, option_type, tol=tol)
        res[SimpleValuationMetrics.IMPLIED_LOGNORMAL_VOL] = sigma_imp

        # 2) compute greeks at implied vol
        greeks = EuropeanOptionAnalytics.european_option_log_normal(
            forward, strike, time_to_expiry, sigma_imp, option_type, calc_risk)

        # 3) compute sensitivities of implied vol using implicit function theorem
        if calc_risk:
            res.update({
                SimpleValuationMetrics.D_LN_VOL_D_FORWARD: \
                    -greeks[SimpleValuationMetrics.DELTA] / greeks[SimpleValuationMetrics.VEGA],
                SimpleValuationMetrics.D_LN_VOL_D_TTE: \
                    -greeks[SimpleValuationMetrics.THETA] / greeks[SimpleValuationMetrics.VEGA],
                SimpleValuationMetrics.D_LN_VOL_D_STRIKE: \
                    -greeks[SimpleValuationMetrics.STRIKE_RISK] / greeks[SimpleValuationMetrics.VEGA]
            })

        return res
    
    @staticmethod
    def implied_normal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut]=CallOrPut.CALL,
        calc_risk : Optional[bool]=False,
        tol: Optional[float]=1e-8
    ) -> Dict:

        res = {}

        # 1) Compute implied normal vol
        sigma_imp = EuropeanOptionAnalytics._implied_normal_vol_bachelier(
            pv, forward, strike, time_to_expiry, option_type, tol=tol)
        res[SimpleValuationMetrics.IMPLIED_NORMAL_VOL] = sigma_imp

        # 2) Compute Greeks at implied vol
        greeks = EuropeanOptionAnalytics.european_option_normal(
            forward, strike, time_to_expiry, sigma_imp, option_type, calc_risk)

        # 3) Compute sensitivities of implied vol
        if calc_risk:
            res.update({
                SimpleValuationMetrics.D_N_VOL_D_FORWARD: \
                    -greeks[SimpleValuationMetrics.DELTA] / greeks[SimpleValuationMetrics.VEGA],
                SimpleValuationMetrics.D_N_VOL_D_TTE: \
                    -greeks[SimpleValuationMetrics.THETA] / greeks[SimpleValuationMetrics.VEGA],
                SimpleValuationMetrics.D_N_VOL_D_STRIKE: \
                    -greeks[SimpleValuationMetrics.STRIKE_RISK] / greeks[SimpleValuationMetrics.VEGA]
            })

        return res

    @staticmethod
    def lognormal_vol_to_normal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        log_normal_sigma: float,
        calc_risk: Optional[bool]=False,
        tol: Optional[float]=1e-8
    ) -> Dict:

        res = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) Black price (log-normal)
        black_res = EuropeanOptionAnalytics.european_option_log_normal(
            forward, strike, time_to_expiry, log_normal_sigma, option_type, calc_risk
        )
        pv = black_res[SimpleValuationMetrics.PV]

        # 2) Implied normal vol (Bachelier)
        res[SimpleValuationMetrics.IMPLIED_NORMAL_VOL] = \
            EuropeanOptionAnalytics._implied_normal_vol_bachelier(
                pv, forward, strike, time_to_expiry, option_type, tol=tol
            )

        if calc_risk:
            # Greeks at implied normal vol
            bachelier_res = EuropeanOptionAnalytics.european_option_normal(
                forward, strike, time_to_expiry, res[SimpleValuationMetrics.IMPLIED_NORMAL_VOL], option_type, True
            )
            # vol risk
            res[SimpleValuationMetrics.D_N_VOL_D_LN_VOL] = black_res[SimpleValuationMetrics.VEGA] / \
                                                            bachelier_res[SimpleValuationMetrics.VEGA]
            # forward risk
            res[SimpleValuationMetrics.D_N_VOL_D_FORWARD] = \
                (black_res[SimpleValuationMetrics.DELTA] \
                 - bachelier_res[SimpleValuationMetrics.DELTA]) / \
                bachelier_res[SimpleValuationMetrics.VEGA]
            # strike risk
            res[SimpleValuationMetrics.D_N_VOL_D_STRIKE] = \
                (black_res[SimpleValuationMetrics.STRIKE_RISK] \
                 - bachelier_res[SimpleValuationMetrics.STRIKE_RISK]) / \
                bachelier_res[SimpleValuationMetrics.VEGA]
            # tte risk
            res[SimpleValuationMetrics.D_N_VOL_D_TTE] = \
                (black_res[SimpleValuationMetrics.THETA] - bachelier_res[SimpleValuationMetrics.THETA]) / \
                bachelier_res[SimpleValuationMetrics.VEGA]

        return res


    @staticmethod
    def normal_vol_to_lognormal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        calc_risk : Optional[bool]=False, 
        tol: Optional[float]=1e-8) -> Dict:

        res = {}

        call_or_put = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) bachelier
        bachelier_res = EuropeanOptionAnalytics.european_option_normal(
            forward, strike, time_to_expiry, normal_sigma, call_or_put, calc_risk)
        pv = bachelier_res[SimpleValuationMetrics.PV]

        # 2) implied log normal vol
        res[SimpleValuationMetrics.IMPLIED_LOGNORMAL_VOL] = \
            EuropeanOptionAnalytics._implied_lognormal_vol_black(
                pv, forward, strike, time_to_expiry, call_or_put, tol=tol)

        # risk 
        if calc_risk:
            black_res = EuropeanOptionAnalytics.european_option_log_normal(
                forward, strike, time_to_expiry, res[SimpleValuationMetrics.IMPLIED_LOGNORMAL_VOL], call_or_put, True)
            # vol risk
            res[SimpleValuationMetrics.D_LN_VOL_D_N_VOL] = \
                bachelier_res[SimpleValuationMetrics.VEGA] / black_res[SimpleValuationMetrics.VEGA]
            # forward risk
            res[SimpleValuationMetrics.D_LN_VOL_D_FORWARD] = \
                (bachelier_res[SimpleValuationMetrics.DELTA] \
                - black_res[SimpleValuationMetrics.DELTA]) / \
                black_res[SimpleValuationMetrics.VEGA]
            # strike risk
            res[SimpleValuationMetrics.D_LN_VOL_D_STRIKE] = \
                (bachelier_res[SimpleValuationMetrics.STRIKE_RISK] \
                - black_res[SimpleValuationMetrics.STRIKE_RISK]) / \
                black_res[SimpleValuationMetrics.VEGA]
            # tte risk
            res[SimpleValuationMetrics.D_LN_VOL_D_TTE] = (
                bachelier_res[SimpleValuationMetrics.THETA]
                - black_res[SimpleValuationMetrics.THETA]
            ) / black_res[SimpleValuationMetrics.VEGA]

        return res

    ### utilities below

    @staticmethod
    def _implied_lognormal_vol_black(
        pv : float,
        forward : float,
        strike : float,
        time_to_expiry : float,
        option_type : Optional[CallOrPut]=CallOrPut.CALL,
        tol : Optional[float]=1e-8,
        vol_min : Optional[float]=1e-6,
        vol_max : Optional[float]=2.,
        max_iter : Optional[int]=1000) -> float:

        # arbitrage bounds
        intrinsic = max(0.0, forward - strike) if option_type == CallOrPut.CALL else max(0.0, strike - forward)
        if pv < intrinsic:
            raise ValueError("Price below intrinsic value")

        # initial guess
        sigma = EuropeanOptionAnalytics._initial_log_normal_implied_vol_guess(forward, time_to_expiry, pv)

        # bisection + newton
        for _ in range(max_iter):

            res = EuropeanOptionAnalytics.european_option_log_normal(
                forward, strike, time_to_expiry, sigma, option_type, True)
            pv_est = res[SimpleValuationMetrics.PV]
            vega = res[SimpleValuationMetrics.VEGA]
            diff = pv_est - pv

            if abs(diff) < tol:
                return sigma
            # newton step only if stable
            if vega > 1e-8 and 0 < sigma - diff/vega < vol_max:
                sigma -= diff / vega
            else:
                # bisection fallback
                sigma = 0.5 * (vol_min + vol_max)

            if pv_est > pv:
                vol_max = sigma
            else:
                vol_min = sigma

        raise RuntimeError("Implied volatility did not converge")

    @staticmethod
    def _implied_normal_vol_bachelier(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut]=CallOrPut.CALL,
        tol: Optional[float]=1e-8,
        vol_min: Optional[float]=1e-8,
        vol_max: Optional[float]=0.1,
        max_iter: Optional[int]=100) -> float:

        # arbitrage bounds
        intrinsic = max(0.0, forward - strike) if option_type == CallOrPut.CALL else max(0.0, strike - forward)
        if pv < intrinsic:
            raise ValueError("Price below intrinsic value")

        # initial guess
        sigma = EuropeanOptionAnalytics._initial_normal_implied_vol_guess(time_to_expiry, pv)

        # bisection + newton
        for _ in range(max_iter):

            res = EuropeanOptionAnalytics.european_option_normal(forward, strike, time_to_expiry, sigma, option_type, True)
            pv_est = res[SimpleValuationMetrics.PV]
            vega = res[SimpleValuationMetrics.VEGA]
            diff = pv_est - pv

            if abs(diff) < tol:
                return sigma
            # newton step only if stable
            if vega > 1e-8 and 0 < sigma - diff/vega < vol_max:
                sigma -= diff / vega
            else:
                # bisection fallback
                sigma = 0.5 * (vol_min + vol_max)
                
            if pv_est > pv:
                vol_max = sigma
            else:
                vol_min = sigma

        raise RuntimeError("Implied normal volatility did not converge")

    @staticmethod
    def _initial_log_normal_implied_vol_guess(forward : float, time_to_expiry : float, pv : float):
        return math.sqrt(2 * math.pi / time_to_expiry) * pv / forward
    
    @staticmethod
    def _initial_normal_implied_vol_guess(time_to_expiry: float, pv: float):
        return pv * math.sqrt(2 * math.pi / time_to_expiry)
    




