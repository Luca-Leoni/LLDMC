###############################
#
#   Classes for using the 
#   Metropolis-Hasting algorithm
#   in order to sample from a 
#   general distribution
#
###############################

###############################
#
#   DEPENDENCIES
#
###############################

from array import array
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import rand

###############################
#
#   SAMPLER CLASS
#
###############################

class LLDMC_MC_Sampler:
    """
        Class that use Metropolis-Hasting algorithm to perform a sampling form a general function
    """

    ###############################
    #   DATA DICTIONARY
    ###############################

    samples    = None           # Array with the centers of the bin in wich the function domain is discretized      np.array(float)
    target_fun = None           # Array with target function evaluated on bin cneters                               np.array(float)
    bin_width  = -1             # Lenght of the bin for the discretization                                          float
    sample     = None           # Index of last random sample evaluated in the run                                  int

    tot_draws  = 0              # total number of draws done                                                        int
    acc_draws  = 0              # number of accepted draws                                                          int
    
    ###############################
    #   CONSTRUCTOR
    ###############################

    def __init__(self, Function, domain: array, bin_width: float = 1E-2) -> None:
        self.set_sampler(Function, domain, bin_width)

    ###############################
    #   FUNCTIONS
    ###############################

    #----------PUBBLIC----------

    def set_sampler(self, Function, domain: array, bin_width: float = 1E-2) -> None:
        """
            description
            ===========
            set the specific of the sampler, in particular set the domain of sampling and the target distribution function

            inputs
            ======
            Function:       target distribution function

            domain:         array with two entries [beg of domain, end of domain]

            bin_width:      define the width of the bins for the discretization of the domain
        """
        self.__set_bin_sampling(domain, bin_width)
        self.target_fun = Function(self.samples)

        self.__thermalyze()

    def draw_sample(self, size: int, div_size: int = 300000, verbose: bool = False) -> float:
        """
            description
            ===========
            draw a sample from the disctribution defined previusly inside the domain selected

            inputs
            ======
            size:           size of the draw that we want to have, like 10 will give 10 draws
        """
        draws = np.array([self.sample])

        if verbose:
            print("Start sampling {:6n} samples...".format(size))

        for ind in range(size):
            i = np.random.randint(0, self.samples.size)
            accept_prob = self.target_fun[i]/self.target_fun[draws[-1]]

            if accept_prob > 1:
                draws = np.append(draws, i)

                self.acc_draws += 1
            else:
                if np.random.uniform() < accept_prob:
                    draws = np.append(draws, i)

                    self.acc_draws += 1
                else:
                    draws = np.append(draws, draws[-1])

            if verbose and ind % 100000 == 0:
                print("Extracted {:6n}-th sample".format(ind))

        self.sample = draws[-1]

        self.tot_draws += size

        return self.samples[draws]

    def plot_distribution(self, n_sample: int = 200000) -> None:
        """
            description
            ===========
            plot the distribution obtained from the function inserted.

            inputs
            ======
            n_sample:       number of sample to draw in order to plot it
        """
        n_hits = self.draw_sample(n_sample, verbose=True)

        sns.displot(n_hits, bins=np.linspace(self.samples[0] - self.bin_width/2, self.samples[-1] + self.bin_width/2, np.size(self.samples)+1), kde=True)
        plt.show()
        

    #----------PRIVATE----------

    def __set_bin_sampling(self, domain: array, bin_width: float) -> None:
        """
            description
            ===========
            function to set the domain and discretize it in several bins.

            inputs
            ======
            domain:         array with two entries [beg of domain, end of domain]

            bin-width:      define the width of the bins for the discretization of the domain
        """
        n_bins = int((domain[1] - domain[0])/bin_width)

        self.bin_width = bin_width
        self.samples   = np.zeros(n_bins)

        for i in range(n_bins):
            self.samples[i] += bin_width*( i + 0.5)

    
    def __thermalyze(self, n_step: int = 10000) -> None:
        """
            description
            ===========
            function to thermalyze the Markov chain, basically takes samples varius time to reach convergence

            inputs
            ======
            n_step:         number of sample to draw
        """
        draws = np.random.randint(0, self.samples.size, n_step)

        # take the first value
        self.sample = draws[0]
        np.delete(draws, 0)

        # Perform Hasting algorithm
        for i in draws:
            accept_prob = self.target_fun[i]/self.target_fun[self.sample]

            if accept_prob > 1:
                self.sample = i
            else:
                if np.random.uniform() > accept_prob:
                    self.sample = i