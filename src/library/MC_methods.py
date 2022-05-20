###############################
#
#   MC lybrari with the main 
#   functions for the program
#
###############################

###############################
#   DEPENDENCIES
###############################

from src.classes.LLMDMC_CDF_Sampler import *
from src.classes.LLMDMC_MC_Sampler import *
import numpy as np

###############################
#   FUNCTIONS
###############################

def simple_integration(Function, domain, delta: float = 1E-4):
    """
        description
        ===========
        Help function to qiuckly evaluate normalization constant of distributions

        inputs
        ======
        Function:       The function needed to integrate

        domain:         array_like object with two entries giving start and end of the domain of integration

        delta:          discretizionation of the domain to use
    """
    result   = 0
    N_bin    = int((domain[1]-domain[0])/delta)

    for i in range(1, N_bin):
        x0 = domain[0] + delta*(i-1)
        x1 = domain[0] + delta*i

        result += 0.5*(Function(x0) + Function(x1))*delta
    
    return result


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
        return (domain[1] - domain[0])*np.mean(Values), (domain[1] - domain[0])*np.std(Values)/np.sqrt(n_samples)
    else:
        return (domain[1] - domain[0])*np.mean(Values)


def MC_1D_integration(Function, domain, prob_dens, n_samples: int = 1000000, err: bool = False, markov_chain: bool = False, n_division: int = 100) -> float:
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
    #---Setup variables---
    result = 0

    if markov_chain:
        sampler = LLDMC_MC_Sampler(prob_dens, domain, 0.001)
    else:
        sampler = LLDMC_CDF_Sampler(prob_dens, domain, 0.001)

    #---Random sampling---
    for i in range(int(n_samples/n_division)):
        result += np.sum(Function(sampler.draw_sample(n_division)))

    if n_samples % n_division != 0:
        result += np.sum(Function(sampler.draw_sample(n_samples % n_division)))

    #---Selction of normalization constant---
    if markov_chain:
        norm_const = simple_integration(prob_dens, domain)
    else:
        norm_const = sampler.norm_const

    #---OUTPUT---
    if not err:
        return result*norm_const/n_samples
    else:
        result = result*norm_const/n_samples

        return result, np.sqrt(np.abs(MC_1D_integration(lambda x: Function(x)**2, domain, prob_dens, n_samples, err=False, markov_chain=markov_chain) - result**2)/n_samples)