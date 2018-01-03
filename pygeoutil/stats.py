import os
import pdb
import numpy as np


def remove_nans(a, b):
    """
    Remove nans from two arrays if there is a nan in the same position in either of the arrays
    :param a:
    :param b:
    :return:
    """
    a = np.asarray(a)
    b = np.asarray(b)

    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]

    return a, b


def remove_nans3(a, b, c):
    """
    Remove nans from three arrays if there is a nan in the same position in either of the other arrays
    :param a:
    :param b:
    :param c:
    :return:
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    mask = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c)
    a = a[mask]
    b = b[mask]
    c = c[mask]

    return a, b, c


def remove_nans4(a, b, c, d):
    """
    Remove nans from four arrays if there is a nan in the same position in either of the other arrays
    :param a:
    :param b:
    :param c:
    :return:
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)

    mask = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c) & ~np.isnan(d)
    a = a[mask]
    b = b[mask]
    c = c[mask]
    d = d[mask]

    return a, b, c, d


def ols(sim, obs, weights=1.0):
    """

    :param sim: predicted (or simulated)
    :param obs: observed, expected
    :param weights:
    :return:
    """
    import statsmodels.api as sm
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    obs, sim = remove_nans(obs, sim)

    results = sm.WLS(sim, sm.add_constant(obs), weights=weights, missing='drop').fit()

    return results


def rmse(sim, obs):
    """
    root mean square error
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """
    sim = np.asarray(sim)
    obs = np.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return np.sqrt(np.mean((sim - obs)**2))


def mae(sim, obs):
    """
    mean absolute error
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = np.asarray(sim)
    obs = np.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return np.mean(abs(sim - obs))


def bias(sim, obs):
    """
    Bias
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = np.asarray(sim)
    obs = np.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return np.mean(sim - obs)


def per_bias(sim, obs):
    """
    Percentage bias
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = np.asarray(sim)
    obs = np.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return 100.0 * np.sum(sim - obs)/np.sum(obs)


def nash_sutcliffe(sim, obs):
    """
    Nash sutcliff efficiency
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = np.asarray(sim)
    obs = np.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return 1 - np.sum((sim - obs)**2)/np.sum((obs - np.mean(obs))**2)


def sign(x, n):
    """

    :param x:
    :param n:
    :return:
    """
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    return s


def mk_test(x, alpha=0.05):
    """
    @author: Michael Schramm
    https://github.com/mps9506/Mann-Kendall-Trend/blob/master/mk_test.py
    This function is derived from code originally posted by Sat Kumar Tomer
    (satkumartomer@gmail.com)
    See also: http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm
    The purpose of the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert
    1987) is to statistically assess if there is a monotonic upward or downward
    trend of the variable of interest over time. A monotonic upward (downward)
    trend means that the variable consistently increases (decreases) through
    time, but the trend may or may not be linear. The MK test can be used in
    place of a parametric linear regression analysis, which can be used to test
    if the slope of the estimated linear regression line is different from
    zero. The regression analysis requires that the residuals from the fitted
    regression line be normally distributed; an assumption not required by the
    MK test, that is, the MK test is a non-parametric (distribution-free) test.
    Hirsch, Slack and Smith (1982, page 107) indicate that the MK test is best
    viewed as an exploratory analysis and is most appropriately used to
    identify stations where changes are significant or of large magnitude and
    to quantify these findings.
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics
    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05)
    """
    from scipy.stats import norm

    n = len(x)

    # calculate S
    s = sign(x, n)

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else:  # there are some ties in data
        tp = np.bincount(np.searchsorted(unique_x, x))
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z
