from enum import Enum
import numpy as np
from typing import Optional, Dict, Any, Tuple
from fixedincomelib.analytics.european_options import SimpleMetrics, EuropeanOptionAnalytics

class SabrMetircs(Enum):

    # parameters
    ALPHA = 'alpha'
    
    # (alpha, beta, nu, rho, forward, strike, tte) => \sigma_k
    D_LN_SIGMA_D_FORWARD = 'd_ln_sigma_d_forward'
    D_LN_SIGMA_D_STRIKE = 'd_ln_sigma_d_strike' 
    D_LN_SIGMA_D_TTE = 'd_ln_sigma_d_tte'
    D_LN_SIGMA_D_ALPHA = 'd_ln_sigma_d_alpha'
    D_LN_SIGMA_D_BETA = 'd_ln_sigma_d_beta'
    D_LN_SIGMA_D_NU = 'd_ln_sigma_d_nu'
    D_LN_SIGMA_D_RHO = 'd_ln_sigma_d_rho'

    # (\sigma_ln_atm, f, tte, beta, nu, rho) => alpha
    D_ALPHA_D_LN_SIGMA_ATM = 'd_alpha_d_ln_sigma_atm'
    D_ALPHA_D_FORWARD = 'd_alpha_d_forward'
    D_ALPHA_D_TTE = 'd_alpha_d_tte'
    D_ALPHA_D_BETA = 'd_alpha_d_beta'
    D_ALPHA_D_NU = 'd_alpha_d_nu'
    D_ALPHA_D_RHO = 'd_alpha_d_rho'

    # (alpha, beta, nu, rho, f, tte) => \sigma_n_atm
    D_NORMAL_SIGMA_D_ALPHA = 'd_normal_sigma_d_alpha'
    D_NORMAL_SIGMA_D_BETA = 'd_normal_sigma_d_beta'
    D_NORMAL_SIGMA_D_NU = 'd_normal_sigma_d_nu'
    D_NORMAL_SIGMA_D_RHO = 'd_normal_sigma_d_rho'
    D_NORMAL_SIGMA_D_FORWARD = 'd_normal_sigma_d_forward'
    D_NORMAL_SIGMA_D_TTE = 'd_normal_sigma_d_tte'
    D_ALPHA_D_NORMAL_SIGMA_ATM = 'd_alpha_d_normal_sigma_atm'

    @classmethod
    def from_string(cls, value: str) -> 'SabrMetircs':
        if not isinstance(value, str):
            raise TypeError('value must be a string')
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(f'Invalid token: {value}') from e

    def to_string(self) -> str:
        return self.value

class SABRAnalytics:

    @staticmethod
    def lognormal_vol_from_alpha(
        forward: float,
        strike: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift : Optional[float]=0.,
        calc_risk: Optional[bool]=False) -> Dict[SabrMetircs|SimpleMetrics, float]:
        
        res: Dict[Any, float] = {}

        ln_sigma, risks = SABRAnalytics._vol_and_risk(
            forward + shift, strike + shift, time_to_expiry, alpha, beta, rho, nu, calc_risk)
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = ln_sigma
        
        if len(risks) == 0:
            return res

        res.update(risks)
        return res

    @staticmethod
    def alpha_from_atm_lognormal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_lognormal: float,
        beta: float,
        rho: float,
        nu: float,
        shift : Optional[float]=0.,
        calc_risk: Optional[bool]=False,
        max_iter: Optional[int] = 50,
        tol: Optional[float]=1e-12) -> Dict[SabrMetircs, float]:
        
        if forward + shift <= 0.:
            raise ValueError('forward must be > 0')
        if time_to_expiry < 0.:
            raise ValueError('time_to_expiry must be >= 0')
        if sigma_atm_lognormal <= 0.:
            raise ValueError('sigma_atm_lognormal must be > 0')
        if abs(rho) >= 1.0:
            raise ValueError('rho must be in (-1,1)')
        if nu < 0.:
            raise ValueError('nu must be >= 0')
        if not (0.0 <= beta <= 1.0):
            raise ValueError('beta should be in [0,1] for standard SABR usage')

        # newton + bisec fallback
        # root finding
        # f = F(alpha, theta) - ln_sigma = 0
        # where F is lognormal_vol_from_alpha
        # alpha^* = alpha(ln_sigma, theta)
        
        this_res = None
        alpha = sigma_atm_lognormal * (forward + shift)**(1. - beta)
        for _ in range(max_iter):
            
            this_res = SABRAnalytics.lognormal_vol_from_alpha( \
                forward, 
                forward, 
                time_to_expiry, 
                alpha, 
                beta, 
                rho, 
                nu, 
                shift, 
                True)
            
            fval = this_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] - sigma_atm_lognormal
            dval = this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA] # df/dAlpha
            
            if abs(fval) < tol:
                break
            
            if dval == 0.0 or not np.isfinite(dval):
                alpha = max(1e-16, alpha * 0.9)
                continue

            alpha_new = alpha - fval / dval

            if (alpha_new <= 0.0) or (not np.isfinite(alpha_new)):
                alpha_new = alpha * 0.5
            if alpha_new > 10.0 * alpha:
                alpha_new = 0.5 * (alpha_new + alpha)

            alpha = alpha_new

        else:
            raise RuntimeError("alpha_from_atm_lognormal_sigma: Newton did not converge")

        res: Dict[SabrMetircs, float] = {SabrMetircs.ALPHA: alpha}

        if not calc_risk:
            return res

        # dalphad...
        # alpha^* = alpha(ln_sigma, theta, target_ln_sigma)
        # F(alpha(ln_sigma, theta), theta) = target_ln_sigma
        # using implicit function theorem
        # df/dalpha * dalpha/dln_sigma = 1 =>             dalpha / dln_sigma = 1 / df/dalpha
        # df/dalpha * dalpha/dtheta  + df/dtheta = 0 =>  dalpha / dtheta = - df/dtheta / df/dalpha

        res[SabrMetircs.D_ALPHA_D_LN_SIGMA_ATM] = 1. / this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA]
        res[SabrMetircs.D_ALPHA_D_BETA] = - this_res[SabrMetircs.D_LN_SIGMA_D_BETA] / this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA]
        res[SabrMetircs.D_ALPHA_D_NU] = - this_res[SabrMetircs.D_LN_SIGMA_D_NU] / this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA]
        res[SabrMetircs.D_ALPHA_D_RHO] = - this_res[SabrMetircs.D_LN_SIGMA_D_RHO] / this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA]
        res[SabrMetircs.D_ALPHA_D_FORWARD] = - this_res[SabrMetircs.D_LN_SIGMA_D_FORWARD] / this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA]
        res[SabrMetircs.D_ALPHA_D_TTE] = - this_res[SabrMetircs.D_LN_SIGMA_D_TTE] / this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA]

        return res

    @staticmethod
    def alpha_from_atm_normal_sigma(
        forward: float,
        time_to_expiry: float,
        sigma_atm_normal: float,
        beta: float,
        rho: float,
        nu: float,
        shift : Optional[float]=0.,
        calc_risk: bool = False,
        max_iter: int = 50,
        tol: float = 1e-8) -> Dict[SabrMetircs, float]:
        
        # at atm, from nv vol to ln vol 
        this_res = EuropeanOptionAnalytics.normal_vol_to_lognormal_vol(
            forward, 
            forward,
            time_to_expiry, 
            sigma_atm_normal, 
            calc_risk, 
            shift,
            tol) 
        
        # compute implied log normal vol
        that_res = SABRAnalytics.alpha_from_atm_lognormal_sigma(
            forward,
            time_to_expiry, 
            this_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL],
            beta, 
            rho, 
            nu,
            shift,
            calc_risk,
            max_iter, 
            tol)
        
        # risk aggregation
        final_res = {SabrMetircs.ALPHA : that_res[SabrMetircs.ALPHA]}

        if calc_risk:
            # dalpha / dsigma
            final_res[SabrMetircs.D_ALPHA_D_NORMAL_SIGMA_ATM] = \
                that_res[SabrMetircs.D_ALPHA_D_LN_SIGMA_ATM] * this_res[SimpleMetrics.D_LN_VOL_D_N_VOL]
            # dalpha / dbeta/nu/rho just copy over
            final_res[SabrMetircs.D_ALPHA_D_BETA] = that_res[SabrMetircs.D_ALPHA_D_BETA]
            final_res[SabrMetircs.D_ALPHA_D_RHO] = that_res[SabrMetircs.D_ALPHA_D_RHO]
            final_res[SabrMetircs.D_ALPHA_D_NU] = that_res[SabrMetircs.D_ALPHA_D_NU]
            # dalpha / dtte
            final_res[SabrMetircs.D_ALPHA_D_TTE] = that_res[SabrMetircs.D_ALPHA_D_TTE] \
                + that_res[SabrMetircs.D_ALPHA_D_LN_SIGMA_ATM] * this_res[SimpleMetrics.D_LN_VOL_D_TTE]
            # dalpha / dforward
            final_res[SabrMetircs.D_ALPHA_D_FORWARD] = that_res[SabrMetircs.D_ALPHA_D_FORWARD] \
                + that_res[SabrMetircs.D_ALPHA_D_LN_SIGMA_ATM] * ( \
                    this_res[SimpleMetrics.D_LN_VOL_D_FORWARD] + this_res[SimpleMetrics.D_LN_VOL_D_STRIKE])

        return final_res
    
    @staticmethod
    def atm_normal_sigma_from_alpha(
        forward: float,
        time_to_expiry: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift : Optional[float]=0.0,
        calc_risk: Optional[bool]=False,
        tol: Optional[float]=1e-8):
        
        # at atm, from alpha to log normal vol 
        this_res = SABRAnalytics.lognormal_vol_from_alpha(
            forward,
            forward, 
            time_to_expiry,
            alpha,
            beta,
            rho,
            nu,
            shift,
            calc_risk)

        # compute normal vol
        that_res = EuropeanOptionAnalytics.lognormal_vol_to_normal_vol(
            forward,
            forward, 
            time_to_expiry, 
            this_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL], 
            calc_risk, 
            shift,
            tol)
        
        # calc risk
        # risk aggregation
        final_res = {SimpleMetrics.IMPLIED_NORMAL_VOL : that_res[SimpleMetrics.IMPLIED_NORMAL_VOL]}
        if calc_risk:
            # dnvol / dalpha|beta|nu|rho 
            final_res[SabrMetircs.D_NORMAL_SIGMA_D_ALPHA] = \
                that_res[SimpleMetrics.D_N_VOL_D_LN_VOL] * this_res[SabrMetircs.D_LN_SIGMA_D_ALPHA]
            final_res[SabrMetircs.D_NORMAL_SIGMA_D_BETA] = \
                that_res[SimpleMetrics.D_N_VOL_D_LN_VOL] * this_res[SabrMetircs.D_LN_SIGMA_D_BETA]
            final_res[SabrMetircs.D_NORMAL_SIGMA_D_RHO] = \
                that_res[SimpleMetrics.D_N_VOL_D_LN_VOL] * this_res[SabrMetircs.D_LN_SIGMA_D_RHO]
            final_res[SabrMetircs.D_NORMAL_SIGMA_D_NU] = \
                that_res[SimpleMetrics.D_N_VOL_D_LN_VOL] * this_res[SabrMetircs.D_LN_SIGMA_D_NU]
            # dnvol / dforward
            final_res[SabrMetircs.D_NORMAL_SIGMA_D_FORWARD] = \
                that_res[SimpleMetrics.D_N_VOL_D_LN_VOL] * \
                    (this_res[SabrMetircs.D_LN_SIGMA_D_FORWARD] + \
                     this_res[SabrMetircs.D_LN_SIGMA_D_STRIKE]) + \
                        that_res[SimpleMetrics.D_N_VOL_D_FORWARD] + \
                        that_res[SimpleMetrics.D_N_VOL_D_STRIKE]
            # dnvol / dtte
            final_res[SabrMetircs.D_NORMAL_SIGMA_D_TTE] = \
                that_res[SimpleMetrics.D_N_VOL_D_LN_VOL] * this_res[SabrMetircs.D_LN_SIGMA_D_TTE] + \
                that_res[SimpleMetrics.D_N_VOL_D_TTE]
            
        return final_res

    @staticmethod
    def _vol_and_risk(F, K, T, a, b, r, n, calc_risk = False, atm_log_fk_cut=1e-12) -> Tuple[float, Dict[SabrMetircs, float]]:

        logFK = np.log(F / K)
        greeks : Dict[SabrMetircs, float] = {}

        # atm case
        if abs(logFK) < atm_log_fk_cut:
            
            Fp = F**(1.0 - b)

            C1 = (1 - b)**2 * a*a / (24 * Fp * Fp)
            C2 = r * b * n * a / (4 * Fp)
            C3 = (2 - 3 * r * r) * n * n / 24
            C  = C1 + C2 + C3
            sigma = (a / Fp) * (1 + C * T)
            
            if calc_risk:
                greeks[SabrMetircs.D_LN_SIGMA_D_ALPHA] = sigma / a + (a * T / Fp)*(2 * C1 / a + C2 / a)
                greeks[SabrMetircs.D_LN_SIGMA_D_NU] = (a * T / Fp) * (C2 / n + 2 * C3 / n)
                greeks[SabrMetircs.D_LN_SIGMA_D_RHO] = (a * T / Fp) * (b * n * a/ (4 * Fp) - r * n * n / 4)
                greeks[SabrMetircs.D_LN_SIGMA_D_BETA] = sigma * np.log(F) - (a * T / Fp) * ((2 * (1 - b) * C1 + C2) * np.log(F))
                greeks[SabrMetircs.D_LN_SIGMA_D_FORWARD] = -(1 - b) / F * sigma - (a * T / Fp) * ((2 * (1 - b) * C1 + C2) / F)
                greeks[SabrMetircs.D_LN_SIGMA_D_TTE] = (a / Fp) * C
                greeks[SabrMetircs.D_LN_SIGMA_D_STRIKE] = 0.

            return sigma, greeks

        # non atm
        fk = (F * K)**((1 - b) / 2)
        z = (n / a) * fk * logFK
        g = np.sqrt(1 - 2 * r * z + z * z)
        x = np.log((g + z - r)/(1 - r))
        A = a / fk
        B = z / x
        C1 = (1-b)**2 * a * a / (24 * fk * fk)
        C2 = r * b * n * a / (4 * fk)
        C3 = (2 - 3 * r * r) * n * n / 24
        C  = C1 + C2 + C3
        D  = 1 + C * T
        sigma = A * B * D

        if calc_risk:

            dz = {
                SabrMetircs.D_LN_SIGMA_D_ALPHA : - z / a,
                SabrMetircs.D_LN_SIGMA_D_NU : z / n,
                SabrMetircs.D_LN_SIGMA_D_BETA : -0.5 * z * (np.log(F) + np.log(K)),
                SabrMetircs.D_LN_SIGMA_D_FORWARD : z * ((1 - b) / (2 * F) + 1 / (F * logFK)),
                SabrMetircs.D_LN_SIGMA_D_STRIKE : z * ((1 - b) / (2 * K) - 1 / (K * logFK))
            }

            dx_dz = 1 / g
            dx_dr = (-z / g - 1.0) / (g + z - r) + 1.0 / (1.0 - r)

            greeks = {}

            for p in [SabrMetircs.D_LN_SIGMA_D_ALPHA, 
                    SabrMetircs.D_LN_SIGMA_D_BETA,
                    SabrMetircs.D_LN_SIGMA_D_FORWARD,
                    SabrMetircs.D_LN_SIGMA_D_STRIKE,
                    SabrMetircs.D_LN_SIGMA_D_NU]:
                Ap = {
                    SabrMetircs.D_LN_SIGMA_D_ALPHA : 1 / a,
                    SabrMetircs.D_LN_SIGMA_D_BETA : 0.5 * (np.log(F) + np.log(K)),
                    SabrMetircs.D_LN_SIGMA_D_FORWARD : - (1 - b) / (2 * F),
                    SabrMetircs.D_LN_SIGMA_D_STRIKE :  - (1 - b) / (2 * K)
                }.get(p, 0.0)

                zp = dz[p]
                xp = dx_dz * zp

                Bp = zp/z - xp/x

                Cp = {
                    SabrMetircs.D_LN_SIGMA_D_ALPHA : 2 * C1 / a + C2 / a,
                    SabrMetircs.D_LN_SIGMA_D_NU : C2/n + 2*C3/n
                }.get(p, 0.0)

                Dp = (T/(1 + C*T))*Cp

                greeks[p] = sigma*(Ap + Bp + Dp)

            # rho
            greeks[SabrMetircs.D_LN_SIGMA_D_RHO] = sigma * (
                -dx_dr / x + (T / (1 + C * T)) * (b * n * a/(4 * fk) - r * n * n / 4))

            # time
            greeks[SabrMetircs.D_LN_SIGMA_D_TTE] = sigma * C / (1 + C * T)

        return sigma, greeks