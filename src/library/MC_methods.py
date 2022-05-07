###############################
#
#   MC lybrari with the main 
#   functions for the program
#
###############################

###############################
#   DEPENDENCIES
###############################

from random import sample
from src.classes.Sampler import *
import numpy as np

###############################
#   FUNCTIONS
###############################

def pi_estimation_1D_integration(n_points: int, error: bool = False) -> float:
    """
        description
        ===========
        Simple function for the estimation of pi, using a 1D MC interation,
        using uniform distribution, of the function sqrt(1 - x^2) inside [0, 1]

        inputs
        ======
        n_points:       total number of points sampled form unifor distribution
    """
    samples = np.random.uniform(0, 1, n_points)
    samples = np.sqrt(1 - np.power(samples, 2))

    if error:
        return 4*np.mean(samples), 4*np.std(samples)
    else:
        return 4*np.mean(samples)


def pi_estimation_2D_integration(n_points: int, error: bool = False) -> float:
    """
        description
        ===========
        Simple function for the estimation of pi, using a 2D MC interation,
        using uniform distribution, of the function f = 1 in unit circle

        inputs
        ======
        n_points:       total number of points sampled form unifor distribution
    """
    samples = np.random.uniform(0, 1, (2, n_points))
    samples = (np.power(samples[0], 2) + np.power(samples[1], 2)) < 1
    samples = np.where(samples, 1, 0)

    if error:
        return 4*np.mean(samples), 4*np.std(samples)
    else:
        return 4*np.mean(samples)


def MC_1D_uniform_integration(Function, domain, n_samples: int = 1000000, err: bool = False) -> float:
    """
        description
        ===========
        Simple function to integrate a general 1D function using a MC algorithm based on uniform random sampling

        inputs
        ======
        Function:       The function needed to integrate
        domain:         array_like object with two entries giving start and end of the domain of integration
        n_samples:      samples to draw in order to perform the mean which gives the integral value
        err:            True if you want the standard deviation to be evaluated and reported in a tuple
    """
    Values = Function(np.random.uniform(domain[0], domain[1], n_samples))

    if err:
        return (domain[1] - domain[0])*np.mean(Values), (domain[1] - domain[0])*np.std(Values)
    else:
        return (domain[1] - domain[0])*np.mean(Values)


def MC_1D_integration(Function, domain, prob_dens, n_samples: int = 1000000, err: bool = False) -> float:
    """
        description
        ===========
        Simple function to integrate a general 1D function using a MC algorithm based on random sampling of a selected distribution

        inputs
        ======
        Function:       The function needed to integrate
        domain:         array_like object with two entries giving start and end of the domain of integration
        prob_dens:      function that will get transformed into a prob_dens through CDF algorithm
        n_samples:      samples to draw in order to perform the mean which gives the integral value
        err:            True if you want the standard deviation to be evaluated and reported in a tuple
    """
    sampler = LLDMC_CDF_Sampler(prob_dens, domain, 0.001)

    values = Function(sampler.draw_sample(n_samples))

    if err:
        return sampler.norm_const * np.mean(values), sampler.norm_const * np.std(values)
    else:
        return sampler.norm_const * np.mean(values)