###############################
#
#   Class for generating and 
#   sampling a general distribution
#   from a function
#
###############################

###############################
#
#   DEPENDENCIES
#
###############################

from array import array
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

class LLDMC_CDF_Sampler:
    """
        Class to generate probability density distribution from an input function 
        and use a CDF algorithm to sample it.
    """

    ###############################
    #   DATA DICTIONARY
    ###############################

    samples    = None           # Array with the centers of the bin in wich the function domain is discretized  np.array(float)
    fun_distr  = None           # Array with the cumulative probability distribution evaluated by the class     np.array(float)
    bin_width  = -1             # Lenght of the bin for the discretization                                      float
    norm_const = -1             # Normalization constant of the function in the selected domain                 float

    ###############################
    #   CONSTRUCTOR
    ###############################

    def __init__(self, Function, domain: array, bin_width: float = 1E-2) -> None:
        self.set_sampler(Function, domain, bin_width)

    ###############################
    #   FUNCTIONS
    ###############################

    #----------PUBBLIC----------

    def plot_distribution(self, n_sample: int = 10000) -> None:
        """
            description
            ===========
            plot the distribution obtained from the function inserted.

            inputs
            ======
            n_sample:       number of sample to draw in order to plot it
        """
        n_hits = self.draw_sample(n_sample)

        sns.displot(n_hits, bins=np.linspace(self.samples[0] - self.bin_width/2, self.samples[-1] + self.bin_width/2, np.size(self.samples)+1), kde=True)
        plt.show()

        
        

    def draw_sample(self, size) -> float:
        """
            description
            ===========
            draw a sample from the disctribution defined previusly inside the domain selected

            inputs
            ======
            size:           size of the draw that we want to have, like 10 will give 10 draws
        """
        random_draw = np.random.uniform(0, 1, size)

        for i, r in enumerate(random_draw):
            random_draw[i] = np.where(self.fun_distr < r)[0][-1]

        return self.samples[np.int_(random_draw)]


    def set_sampler(self, Function, domain: array, bin_width: float = 1E-2) -> None:
        """
            description
            ===========
            set the specific of the sampler, in particular set the domain of sampling and the comulative distribution function

            inputs
            ======
            function:       function to use for the evaluation of the distribution

            domain:         array with two entries [beg of domain, end of domain]

            bin-width:      define the width of the bins for the discretization of the domain
        """
        self.__set_bin_sampling(domain, bin_width)
        self.__set_fun_distr(Function)

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


    def __set_fun_distr(self, Function) -> None:
        """
            description
            ===========
            set the distribution used to sample by approximating the comulative probability distribution 
            inside every bin

            inputs
            ======
            function:       function to use for the evaluation of the distribution
        """
        n_size         = np.size(self.samples)
        self.fun_distr = np.zeros(n_size)
        values         = Function(self.samples)*self.bin_width

        for i in range(1, n_size):
            self.fun_distr[i] = self.fun_distr[i-1] + values[i]

        self.norm_const = self.fun_distr[-1]

        self.fun_distr /= self.norm_const


    def __draw_indicies(self, size) -> int:
        """
            description
            ===========
            draw an indices of self.samples from the disctribution defined previusly inside the domain selected

            inputs
            ======
            size:           size of the draw that we want to have, like 10 will give 10 draws
        """
        random_draw = np.random.uniform(0, 1, size)

        for i, r in enumerate(random_draw):
            random_draw[i] = np.where(self.fun_distr < r)[0][-1]

        return np.int_(random_draw)