import pandas as pd
from fixedincomelib.analytics import *

# european option

def qfEuropeanOptionLogNormal(
    forward : float,
    strike : float,
    time_to_expiry : float,
    log_normal_sigma : float,
    option_type : Optional[str]='call',
    calc_risk : Optional[bool]=False):

    res = EuropeanOptionAnalytics.european_option_log_normal(
        forward,
        strike,
        time_to_expiry,
        log_normal_sigma,
        CallOrPut.from_string(option_type),
        calc_risk)
    
    # remove tte risk
    if SimpleMetrics.TTE_RISK in res:
        res.pop(SimpleMetrics.TTE_RISK)

    return res

def qfEuropeanOptionImpliedLogNormalVol(
    pv: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    option_type: Optional[str]='call',
    calc_risk : Optional[bool]=False,
    tol: Optional[float] = 1e-8):

    res = EuropeanOptionAnalytics.implied_lognormal_vol_sensitivities(
        pv,
        forward,
        strike,
        time_to_expiry,
        CallOrPut.from_string(option_type),
        calc_risk,
        tol)

    return res

def qfEuropeanOptionNormal(
    forward : float,
    strike : float,
    time_to_expiry : float,
    log_normal_sigma : float,
    option_type : Optional[str]='call',
    calc_risk : Optional[bool]=False):

    res = EuropeanOptionAnalytics.european_option_normal(
        forward,
        strike,
        time_to_expiry,
        log_normal_sigma,
        CallOrPut.from_string(option_type),
        calc_risk)

    # remove tte risk
    if SimpleMetrics.TTE_RISK in res:
        res.pop(SimpleMetrics.TTE_RISK)

    return res

def qfEuropeanOptionImpliedNormalVol(
    pv: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    option_type: Optional[str]='call',
    calc_risk : Optional[bool]=False,
    tol: Optional[float] = 1e-8):

    res = EuropeanOptionAnalytics.implied_normal_vol_sensitivities(
        pv,
        forward,
        strike,
        time_to_expiry,
        CallOrPut.from_string(option_type),
        calc_risk,
        tol)

    return res

def qfEuropeanOptionNormalVolFromLogNormalVol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    log_normal_sigma: float,
    shift : Optional[float]=0.,
    calc_risk : Optional[bool]=False, 
    tol: Optional[float]=1e-8):

    res = EuropeanOptionAnalytics.lognormal_vol_to_normal_vol(
        forward,
        strike,
        time_to_expiry,
        log_normal_sigma,
        calc_risk,
        shift,
        tol) 
    
    return res

def qfEuropeanOptionLogNormalVolFromNormalVol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    normal_sigma: float,
    shift : Optional[float]=0.,
    calc_risk : Optional[bool]=False, 
    tol: Optional[float]=1e-8):

    res = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(
        forward,
        strike,
        time_to_expiry,
        normal_sigma,
        calc_risk,
        shift,
        tol) 
    
    return res

### sabr 

def qfEuropeanOptionSABRLogNormalSigma(
    forward : float,
    strike : float,
    time_to_expiry : float,
    alpha : float, 
    beta : float,
    rho : float, 
    nu : float,
    shift : Optional[float]=0.,
    calc_risk : Optional[bool]=False):

    res = SABRAnalytics.lognormal_vol_from_alpha(
        forward, strike, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk)
    
    return res

def qfEuropeanOptionSABRAlphaFromATMLogNormalSigma( \
    forward : float,
    time_to_expiry : float,
    sigma_atm_log_normal : float,
    beta : float,
    rho : float, 
    nu : float,
    shift : Optional[float]=0.,
    calc_risk : Optional[bool]=False,
    max_iter : Optional[int]=50,
    tol : Optional[float]=1e-8):
    
    res = SABRAnalytics.alpha_from_atm_lognormal_sigma( \
        forward, time_to_expiry, sigma_atm_log_normal, beta, rho, nu, shift, calc_risk, max_iter, tol)
    
    return res

def qfEuropeanOptionSABRAlphaFromATMNormalSigma( \
    forward : float,
    time_to_expiry : float,
    sigma_atm_normal : float,
    beta : float,
    rho : float, 
    nu : float,
    shift : Optional[float]=0.,
    calc_risk : Optional[bool]=False,
    max_iter : Optional[int]=50,
    tol : Optional[float]=1e-8):

    res = SABRAnalytics.alpha_from_atm_normal_sigma( \
        forward, time_to_expiry, sigma_atm_normal, beta, rho, nu, shift, calc_risk, max_iter, tol)
    
    return res

def qfEuropeanOptionSABRATMNormalSigmaFromAlpha(
    forward : float,
    time_to_expiry : float,
    alpha : float, 
    beta : float,
    rho : float, 
    nu : float,
    shift : Optional[float]=0.,
    calc_risk : Optional[bool]=False,
    tol : Optional[float]=1e-8):

    res = SABRAnalytics.atm_normal_sigma_from_alpha(
        forward, time_to_expiry, alpha, beta, rho, nu, shift, calc_risk, tol)
    
    return res