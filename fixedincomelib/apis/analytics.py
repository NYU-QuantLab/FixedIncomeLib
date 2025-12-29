import pandas as pd
from fixedincomelib.analytics import *

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

    return pd.DataFrame(res.items(), columns=['Name', 'Value'])

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

    return pd.DataFrame(res.items(), columns=['Name', 'Value'])

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

    return pd.DataFrame(res.items(), columns=['Name', 'Value'])

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

    return pd.DataFrame(res.items(), columns=['Name', 'Value'])

def qfEuropeanOptionNormalVolFromLogNormalVol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    log_normal_sigma: float,
    calc_risk : Optional[bool]=False, 
    tol: Optional[float]=1e-8):

    res = EuropeanOptionAnalytics.lognormal_vol_to_normal_vol(
        forward,
        strike,
        time_to_expiry,
        log_normal_sigma,
        calc_risk, 
        tol) 
    
    return pd.DataFrame(res.items(), columns=['Name', 'Value'])

def qfEuropeanOptionLogNormalVolFromNormalVol(
    forward: float,
    strike: float,
    time_to_expiry: float,
    normal_sigma: float,
    calc_risk : Optional[bool]=False, 
    tol: Optional[float]=1e-8):

    res = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(
        forward,
        strike,
        time_to_expiry,
        normal_sigma,
        calc_risk, 
        tol) 
    
    return pd.DataFrame(res.items(), columns=['Name', 'Value'])