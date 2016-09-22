import numpy
import statsmodels.api as sm


def remove_nans(a, b):
    """
    Remove nans from two arrays if there is a nan in the same position in either of the arrays
    :param a:
    :param b:
    :return:
    """
    a = numpy.asarray(a)
    b = numpy.asarray(b)

    mask = ~numpy.isnan(a) & ~numpy.isnan(b)
    a = a[mask]
    b = b[mask]

    return a, b


def ols(sim, obs):
    """

    :param sim: predicted (or simulated)
    :param obs: observed, expected
    :return:
    """

    obs = numpy.asarray(obs)
    sim = numpy.asarray(sim)
    obs, sim = remove_nans(obs, sim)

    results = sm.OLS(sim, sm.add_constant(obs), missing='drop').fit()

    return results


def rmse(sim, obs):
    """
    root mean square error
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """
    sim = numpy.asarray(sim)
    obs = numpy.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return numpy.sqrt(numpy.mean((sim - obs)**2))


def mae(sim, obs):
    """
    mean absolute error
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = numpy.asarray(sim)
    obs = numpy.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return numpy.mean(abs(sim - obs))


def bias(sim, obs):
    """
    Bias
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = numpy.asarray(sim)
    obs = numpy.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return numpy.mean(sim - obs)


def per_bias(sim, obs):
    """
    Percentage bias
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = numpy.asarray(sim)
    obs = numpy.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return 100.0 * numpy.sum(sim - obs)/numpy.sum(obs)


def nash_sutcliffe(sim, obs):
    """
    Nash sutcliff efficiency
    :param sim: simulated data
    :param obs: observed data
    :return gof measure
    """            
    sim = numpy.asarray(sim)
    obs = numpy.asarray(obs)
    obs, sim = remove_nans(obs, sim)

    return 1 - numpy.sum((sim - obs)**2)/numpy.sum((obs - numpy.mean(obs))**2)
