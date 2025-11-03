import numpy as np
def hagan_implied_vol(F, K, T, alpha, beta, rho, nu):
    eps = 1e-12
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return np.nan
    if abs(F - K) < eps:
        term1 = ((1 - beta)**2 * alpha**2) / (24 * (F**(2 - 2*beta)))
        term2 = (rho * beta * alpha * nu) / (4 * (F**(1 - beta)))
        term3 = ((2 - 3 * rho**2) * nu**2) / 24
        return (alpha / (F**(1 - beta))) * (1 + (term1 + term2 + term3) * T)
    logFK = np.log(F / K)
    FK_beta = (F * K)**((1 - beta) / 2.0)
    z = (nu / alpha) * FK_beta * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    denom = 1 + ((1 - beta)**2 / 24.0) * (logFK**2) + ((1 - beta)**4 / 1920.0) * (logFK**4)
    base = alpha / (FK_beta * denom)
    vol = base * (z / x_z)
    adj = 1 + T * (
        ((1 - beta)**2 * alpha**2) / (24 * (FK_beta**2))
        + (rho * beta * nu * alpha) / (4 * FK_beta)
        + ((2 - 3 * rho**2) * nu**2) / 24
    )
    return vol * adj
