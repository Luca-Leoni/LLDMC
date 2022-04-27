###############################
#
#   MC lybrari with the main 
#   functions for the program
#
###############################

###############################
#   DEPENDENCIES
###############################

import numpy as np

###############################
#   SAMPLER CLASS
###############################

class LLDMC_Sampler:
    """
        Class to generate probability density distribution from an input function 
        and use a Metropolis-Hastings algorithm to sample it.
    """

    #---------DATA DICTIONARY---------

    samples   = None        # Array with the centers of the bin in wich the function domain is discretized  np.array(float)
    fun_distr = None        # Array with the probability of every bin generated from the function           np.array(float)
    delta_bin = -1          # Lenght of the bin for the discretization                                      float

    #---------CONSTRUCTOR---------

    def __init__(self, function, domain, ) -> None:
        pass

    #---------FUNCTIONS---------