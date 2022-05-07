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