###############################
#
#   Classes for rapresenting a 
#   general Feymann dyagram
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

class LLDMC_Diagram:
    """
        Class used to rapresent a Feymann dyagram of a general order with only one type 
        of interaction inside it, so that the vertices are defined by a strength V constant.
    """

    ###############################
    #   DATA DICTIONARY
    ###############################

    order = -1          # order of the dyagram          int
    