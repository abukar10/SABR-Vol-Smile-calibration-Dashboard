import numpy as np

def hagan_implied_vol(F, K, T, alpha, beta, rho, nu, eps=1e-12):
    """
    Hagan et al. (2002) SABR implied Black volatility (robust implementation)

    Parameters
    ----------
    F : float
        Forward price
    K : float
        Strike
    T : float
        Time to expiry (year fraction)
    alpha : float
        SABR alpha (volatility level)
    beta : float
        SABR beta (elasticity, 0 <= beta <= 1)
    rho : float
        SABR rho (correlation, -1 < rho < 1)
    nu : float
        SABR nu (vol-of-vol)
    eps : float
        Small epsilon for ATM stability

    Returns
    -------
    float
        Implied Black volatility (SABR approximation)
    """
    # Sanity checks
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return np.nan

    one_minus_beta = 1.0 - beta

    # ----- ATM (F ≈ K) -----
    if abs(F - K) < eps:
        F_beta = F ** one_minus_beta
        term1 = (one_minus_beta**2 / 24.0) * (alpha**2 / F_beta**2)
        term2 = (rho * beta * nu * alpha) / (4.0 * F_beta)
        term3 = ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
        return (alpha / F_beta) * (1.0 + (term1 + term2 + term3) * T)

    # ----- General Case (F ≠ K) -----
    logFK = np.log(F / K)
    FK_beta = (F * K) ** (0.5 * one_minus_beta)

    # Guard against extreme cases
    if FK_beta <= 0 or np.isnan(FK_beta):
        return np.nan

    z = (nu / alpha) * FK_beta * logFK

    # Numerical safety for sqrt term
    sqrt_term = np.sqrt(np.maximum(0.0, 1.0 - 2.0 * rho * z + z * z))
    numerator = sqrt_term + z - rho
    denominator = 1.0 - rho

    # Protect against invalid log args
    if numerator <= 0.0 or denominator <= 0.0:
        z_over_x = 1.0  # safe fallback for degenerate cases
    else:
        x_z = np.log(numerator / denominator)
        if abs(x_z) < 1e-12:
            z_over_x = 1.0
        else:
            z_over_x = z / x_z

    # Denominator correction D(F,K)
    denom = 1.0 + (one_minus_beta**2 / 24.0) * logFK**2 \
                + (one_minus_beta**4 / 1920.0) * logFK**4

    # A(F,K)
    A = alpha / (FK_beta * denom)

    # Time-dependent correction
    term1 = (one_minus_beta**2 / 24.0) * (alpha**2 / (FK_beta**2))
    term2 = (rho * beta * nu * alpha) / (4.0 * FK_beta)
    term3 = ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
    B = 1.0 + (term1 + term2 + term3) * T

    sigma = A * z_over_x * B
    return float(np.maximum(sigma, 0.0))
