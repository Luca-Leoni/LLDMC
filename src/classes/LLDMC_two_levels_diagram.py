###############################
#
#   Class for diagrammtically
#   study the thermodynamical
#   properties of a quantum system
#   of a one particle system of spin-1/2
#   in presence of a x-z magnetic field
#   so the Hamiltonian is
#   
#       H = h*rho_z + G*rho_x
#
###############################

###############################
#
#   DEPENDENCIES
#
###############################

from operator import index
from os import times
import random
from select import select
from time import time
from unittest import result
import numpy as np
from requests import delete

from scipy import rand
from src.library.MC_methods import expo_sampling

###############################
#
#   CLASS
#
###############################

class LLDMC_two_levels_diagram:
    """
        Rappresents one of the two contributions inside the general integral form of Z in a two level system, up propagation or down propagation
    """

    ###############################
    #   DATA DICTIONARY
    ###############################
    
    h             = None        # value of the energy constant inside the Hamiltonian                               float
    G             = None        # value of the energy constant inside the Hamiltonian                               float

    order         = 0           # order of the current diagram we are looking at                                    int
    lenghts       = None        # array with lenghts of the segments of every prop.                                 np.array(float)
    spins         = None        # array with the spins of particle in a segment +1 if up or -1 if down              np.array(int)

    ###############################
    #   CONSTRUCTOR
    ###############################

    def __init__(self,initial_state: int, h: float, G: float, beta: float) -> None:
        """
            inputs
            ======
            initial_state:      value of the initial state +1 if up or -1 if down
            
            h:                  value of the energy constant inside the Hamiltonian

            G:                  value of the energy constant inside the Hamiltonian

            beta:               value of the usual thermodynamic constant 1/kT
        """
        self.h = h
        self.G = G

        self.order   = 0
        self.lenghts = np.array([beta])
        self.spins   = np.array([initial_state])

        for i in range(100000):
            self.MC_step()

    
    ###############################
    #   FUNCTIONS
    ###############################

    #----------PUBBLIC----------

    def MC_step(self):
        """
            description
            ===========
            function to perform one montecarlo step inside the chain, selecting with same probability to add or eliminate a vertex inside the diagram.
        """
        if self.order == 0:
            self.__add_segment()
        else:
            if np.random.uniform() < 0.5:
                self.__add_segment()
            else:
                self.__remove_segment()


    def get_weight(self) -> float:
        """
            description
            ===========
            function to evaluate the weight of the current diagram
        """
        result = self.lenghts.dot(-self.spins*self.h)

        return np.exp(result)*np.power(self.G, self.order)

        

    #----------PRIVATE----------

    def __add_segment(self) -> None:
        """
            description
            ===========
            function to add a segment inside the currect dyagrams, obviusly it hads two vertex since odd order diagrams gives 0.
            So, basically selct at random one piace of the diagram and insert inside it another piece of opposite spin, so two verticies are added.
        """
        # select the segment of interest
        n_segments      = np.size(self.lenghts)
        draw_indx       = random.randint(0, n_segments-1)

        old_lenght      = self.lenghts[draw_indx]
        spin_segment    = self.spins[draw_indx]

        # sample the position of the first vertex
        time1 = expo_sampling(0, old_lenght, spin_segment*self.h)                       # drawing start of new segment
        new_lenght = expo_sampling(0, old_lenght - time1, -spin_segment*self.h)         # drawing lenghts of new segment

        # transition probability (remember n. vertices = n. segments - 1)
        acc_prob = self.G*self.G*np.exp(-spin_segment*self.h*new_lenght)*n_segments/(n_segments + 1)

        # Hasting algorithm
        if acc_prob > 1:
            self.lenghts[draw_indx] = time1
            
            self.lenghts = np.insert(self.lenghts, draw_indx + 1, old_lenght - new_lenght - time1)
            self.lenghts = np.insert(self.lenghts, draw_indx + 1, new_lenght)

            self.spins = np.insert(self.spins, draw_indx + 1, spin_segment)
            self.spins = np.insert(self.spins, draw_indx + 1, -spin_segment)

            self.order += 2
        else: 
            if np.random.uniform() < acc_prob:
                self.lenghts[draw_indx] = time1
            
                self.lenghts = np.insert(self.lenghts, draw_indx + 1, old_lenght - new_lenght - time1)
                self.lenghts = np.insert(self.lenghts, draw_indx + 1, new_lenght)

                self.spins = np.insert(self.spins, draw_indx + 1, spin_segment)
                self.spins = np.insert(self.spins, draw_indx + 1, -spin_segment)

                self.order += 2
    

    def __remove_segment(self) -> None:
        """
            description
            ===========
            function to remove a random segment inside the currect dyagram so to diminuisce the diagram's order by 2.
        """
        # select the segment of interest
        n_segments      = np.size(self.lenghts)
        select          = random.randint(1, n_segments-2)

        # transition probability (remember n. vertices = n. segments - 1)
        acc_prob = np.exp(self.spins[select]*self.h*self.lenghts[select])*(n_segments - 1)/(n_segments*self.G*self.G)

        # Hasting algorithm
        if acc_prob > 1:
            self.lenghts[select-1] = self.lenghts[select] + self.lenghts[select-1] + self.lenghts[select+1]

            self.lenghts = np.delete(self.lenghts, select)
            self.lenghts = np.delete(self.lenghts, select)
            self.spins = np.delete(self.spins, select)
            self.spins = np.delete(self.spins, select)

            self.order -= 2
        else: 
            if np.random.uniform() < acc_prob:
                self.lenghts[select-1] = self.lenghts[select] + self.lenghts[select-1] + self.lenghts[select+1]

                self.lenghts = np.delete(self.lenghts, select)
                self.lenghts = np.delete(self.lenghts, select)
                self.spins = np.delete(self.spins, select)
                self.spins = np.delete(self.spins, select)

                self.order -= 2
