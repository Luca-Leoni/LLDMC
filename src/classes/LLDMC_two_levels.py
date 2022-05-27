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

from cProfile import label
import numpy as np
import timeit
import os
import matplotlib.pyplot as plt
import gc

from src.classes.LLDMC_two_levels_diagram import *

###############################
#
#   OTHER SUPPORT FUNCTION
#
###############################

def analytic_solution_z_magne(h, G, beta):
    D = np.sqrt(h**2 + G**2)

    return -np.tanh(beta*D)*h/D

def analytic_solution_x_magne(h, G, beta):
    D = np.sqrt(h**2 + G**2)

    return -np.tanh(beta*D)*G/D


###############################
#
#   CLASS
#
###############################

class LLDMC_two_levels:
    """
        Class for diagrammtically study the thermodynamical properties of a quantum system of a one particle system of spin-1/2 in presence of a x-z magnetic field so the Hamiltonian is
            H = h*rho_z + G*rho_x
    """

    ###############################
    #   DATA DICTIONARY
    ###############################
    
    up_diagram      = None          # diagram of the particle starting in spin up                                       diagram
    dw_diagram      = None          # diagram of the particle starting in spin down                                     diagram
    h               = None          # value of the energy constant inside the Hamiltonian                               float
    G               = None          # value of the energy constant inside the Hamiltonian                               float
    beta            = None          # array with the value of beta used in computations                                 np.array(float)

    orders          = None          # statistics of the order of the diagrams                                           np.array(int)
    z_magne         = None          # magnetization on the z_axes contribution                                          np.array(float)
    x_magne         = None          # magnetization on the x_axes contribution                                          np.array(float)
    up_contribution = None          # contribution given by the spin up part                                            np.array(float)
    dw_contribution = None          # Contribution given by the spin down part                                          np.array(float)
    simulation_time = None          # time counter for the simulation(always usefull)                                   float

    ###############################
    #   CONSTRUCTOR
    ###############################

    def __init__(self, h: float = 0, G: float = 0, beta: list = [0, 0], beta_sampling: int = 2, n_sample: int = 1000000, path: str = None) -> None:
        """
            description
            ===========
            Construct the simulator using the inputs h, G and beta range. Is also possible to not insert those and insert the path with folder with all specifics.

            inputs
            ======
            initial_state:      value of the initial state +1 if up or -1 if down
            
            h:                  value of the energy constant inside the Hamiltonian

            G:                  value of the energy constant inside the Hamiltonian

            beta:               array with the [starting value of beta, final value of beta]

            beta_sampling:      number of beta values in between the domain selected

            n_sample:           number of sample for the simulation that starts right away

            path:               You can simply specify the path to the folder with all information file and use that (in this case doesn't start simulation right away)
        """
        if path != None:
            self.read_data(path)
        else:
            self.h    = h
            self.G    = G
            self.beta = np.linspace(beta[0], beta[1], beta_sampling)

            self.order           = np.zeros((beta_sampling,15))
            self.z_magne         = np.zeros(beta_sampling)
            self.x_magne         = np.zeros(beta_sampling)
            self.up_contribution = np.zeros(beta_sampling)
            self.dw_contribution = np.zeros(beta_sampling)

            self.simulate(n_sample)

    ###############################
    #   FUNCTIONS
    ###############################

    #----------MAIN----------

    def simulate(self, n_sample = 1000000):
        """
            description
            ===========
            function to effectually sample the diagrams in the expansion and accumulate statistics

            inputs
            ======
            n_sample:       number of samples to be drawn for every up and dw contribution 
        """
        self.simulation_time = timeit.default_timer()
        print("Starting simulation...\n")

        for i, beta in enumerate(self.beta):
            self.up_diagram = LLDMC_two_levels_diagram(1, self.h, self.G, beta)
            self.dw_diagram = LLDMC_two_levels_diagram(-1, self.h, self.G, beta)

            gc.collect()

            print("Simulation for beta: {:10n}      time up to now: {:10n}s".format(beta, timeit.default_timer()-self.simulation_time))

            for j in range(n_sample):
                self.up_diagram.MC_step()
                self.dw_diagram.MC_step()

                self.up_contribution[i] += self.up_diagram.get_weight()
                self.dw_contribution[i] += self.dw_diagram.get_weight()

                self.order[i][int(self.up_diagram.order/2)] += 1
                self.order[i][int(self.dw_diagram.order/2)] += 1

            gc.collect()
            
            self.up_contribution[i] /= n_sample
            self.dw_contribution[i] /= n_sample

        self.simulation_time = timeit.default_timer() - self.simulation_time

        print("\nSimulation finished, starting observable evaluation...                           time up to now: {:10n}".format(self.simulation_time))

        for i, beta in enumerate(self.beta):
            self.z_magne[i] = self.up_contribution[i] - self.dw_contribution[i]

            self.x_magne[i] = 0
            for j, count in enumerate(self.order[i]): 
                self.x_magne[i] += j*2*count
            self.x_magne[i] /= -n_sample*self.G*beta

    #----------PUBBLIC----------

    def write_data(self, path: str):
        """
            description
            ===========
            function to write down information about all the statistics accumulated, it will create a folder in the path sad and insert there all the data

            inputs
            ======
            path:           path of the folder that will be created
        """
        os.system('rm -r ' + path)
        os.mkdir(path)

        # create a specific file
        file = open(path + '/SPECIFIC', 'w')

        file.write('# Simulation time: {:10n}s       Samples drawn per beta: {:10n}\n'.format(self.simulation_time, np.sum(self.order[0])))
        file.write('# Input parameters:\n')
        file.write('{:10n} # h\n'.format(self.h))
        file.write('{:10n} # G\n'.format(self.G))
        file.write('{:10n} # beta_min\n'.format(self.beta[0]))
        file.write('{:10n} # beta_max\n'.format(self.beta[-1]))
        file.write('{:10n} # beta_sampling\n'.format(np.size(self.beta)))

        file.close()
        # create order file informations
        file = open(path + '/ORDER', 'w')

        line = '#'
        for i in range(15):
            line += '{:10n}  '.format(i*2)
        line += ' beta'
        file.write(line + '\n')

        for j, beta in enumerate(self.order):
            line = ''
            for i in beta:
                line += '{:10n}  '.format(i)

            line += '#{:10n}'.format(self.beta[j])
            file.write(line + '\n')
        
        file.close()


        # create z_magnetization file informations
        file = open(path + '/Z_MAGNETIZATION', 'w')

        file.write('# {:10s} {:10s}\n'.format('beta', 'z_magne'))

        for j, magne in enumerate(self.z_magne):
            line = '{:10n} {:10n}'.format(self.beta[j], magne)
            file.write(line + '\n')

        file.close()

        # create x_magnetization file informations
        file = open(path + '/X_MAGNETIZATION', 'w')

        file.write('# {:10s} {:10s}\n'.format('beta', 'z_magne'))

        for j, magne in enumerate(self.x_magne):
            line = '{:10n} {:10n}'.format(self.beta[j], magne)
            file.write(line + '\n')

        file.close()

        # create up_contribution file informations
        file = open(path + '/UP_CONTRIBUTION', 'w')

        file.write('# {:10s} {:10s}\n'.format('beta', 'up_contri'))

        for j, contr in enumerate(self.up_contribution):
            line = '{:10n} {:10n}'.format(self.beta[j], contr)
            file.write(line + '\n')

        file.close()

        # create dw_contribution file informations
        file = open(path + '/DW_CONTRIBUTION', 'w')

        file.write('# {:10s} {:10s}\n'.format('beta', 'dw_contri'))

        for j, contr in enumerate(self.dw_contribution):
            line = '{:10n} {:10n}'.format(self.beta[j], contr)
            file.write(line + '\n')

        file.close()
        

    def read_data(self, path: str):
        """
            description
            ===========
            function to write down information about all the statistics accumulated, it will create a folder in the path sad and insert there all the data

            inputs
            ======
            path:           path of the folder that will be created
        """

        # read spefic
        data = np.loadtxt(path + '/SPECIFIC')

        self.h = data[0]
        self.G = data[1]
        self.beta = np.linspace(data[2], data[3], int(data[4]))

        # read orders
        self.order = np.loadtxt(path + '/ORDER')

        # read z_magne
        data = np.loadtxt(path + '/Z_MAGNETIZATION')

        self.z_magne = data[:,1]

        # read x_magne
        data = np.loadtxt(path + '/X_MAGNETIZATION')

        self.x_magne = data[:,1]

        # read up_contribution
        data = np.loadtxt(path + '/UP_CONTRIBUTION')

        self.up_contribution = data[:,1]

        # read dw_contribution
        data = np.loadtxt(path + '/DW_CONTRIBUTION')

        self.dw_contribution = data[:,1]


    def plot_magne(self):
        plt.plot(self.beta, self.z_magne, label='z: computed')
        plt.plot(self.beta, analytic_solution_z_magne(self.h, self.G, self.beta), label='z: analytic')

        plt.legend()
        plt.show()

        plt.plot(self.beta, self.x_magne, label='x: computed')
        plt.plot(self.beta, analytic_solution_x_magne(self.h, self.G, self.beta), label='x: analytic')

        plt.legend()
        plt.show()

    
    #----------PRIVATE----------

    # def __create_general_heading(self, file: str) -> TextIOWrapper:
    #     """
    #         description
    #         ===========
    #         support function to write down general headings informations
    #     """
    #     file = open(file, 'w')
    #     file.write('# Input parameters:   h = {:10n}   G = {:10n}'.format(self.h, self.G))
    #     file.write('# Simulation time: {:10n}       Samples drawn per beta: {:10n}'.format(self.simulation_time, np.sum(self.order[0])))