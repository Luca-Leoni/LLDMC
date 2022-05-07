###############################
#   DEPENDENCIES
###############################

import matplotlib.pyplot as plt
from src.library.MC_methods import *
from src.classes.Sampler import *

###############################
#   FUNCTIONS
###############################

def linear(x: float) -> float:
    return x

def quadratic(x: float) -> float:
    return x**2 

def goal(x: float) -> float:
    return x*np.exp(-x)

def expon(x: float) -> float:
    return np.exp(-x)


###############################
#   FIRST EXERCISE
###############################


def pi_estimation_race(save: bool = False):
    n_sample = np.array([100, 1000, 10000, 100000, 200000])
    
    pi_1D    = np.zeros(5)
    pi_1D_er = np.zeros(5)
    pi_2D    = np.zeros(5)
    pi_2D_er = np.zeros(5)
    
    for i, n in enumerate(n_sample):
        pi_1D[i], pi_1D_er[i] = pi_estimation_1D_integration(n, True)
        pi_2D[i], pi_2D_er[i] = pi_estimation_2D_integration(n, True)

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(n_sample, np.array(pi_1D), '-o', label="1D mean")
    ax[0].plot(n_sample, np.array(pi_2D), '-o', label="2D mean")

    ax[1].plot(n_sample, np.array(pi_1D_er), '-o', label="1D error")
    ax[1].plot(n_sample, np.array(pi_2D_er), '-o', label="2D error")

    ax[0].legend()
    ax[1].legend()

    if save:
        plt.savefig("Results/Images/piRace.png")
    else:
        plt.show()


###############################
#   SECOND EXERCISE
###############################

def integration_race(Function, domain, prob, goal: float, save: bool = False, name: str = "Integration race: Custom vs Uniform"):
    n_samples = np.array([100, 1000, 10000, 100000, 1000000])
    res_unif  = []
    res_prob  = []

    for n in n_samples:
        res_unif.append(MC_1D_uniform_integration(lambda x: Function(x)*prob(x), domain, n_samples=n, err=True))
        res_prob.append(MC_1D_integration(Function, domain, prob, n_samples=n, err=True))
    
    res_unif = np.array(res_unif) * (1 - np.exp(-5))
    res_prob = np.array(res_prob)

    fig, ax = plt.subplots(2,1)

    fig.suptitle(name, fontsize=16)

    ax[0].plot(n_samples, res_unif[:,0], '-o', label="Mean: Unif")
    ax[0].plot(n_samples, res_prob[:,0], '-o', label="Mean: prob")
    ax[0].axhline(goal, linestyle='--')

    ax[1].plot(n_samples, res_unif[:,1], '-o', label="Err: Unif")
    ax[1].plot(n_samples, res_prob[:,1], '-o', label="Err: prob")

    ax[0].legend()
    ax[1].legend()

    if save:
        plt.savefig("Results/Images/" + name + ".png")
    else:
        plt.show()


###############################
#   MAIN
###############################

if __name__ == '__main__':
    seed = 63
    np.random.seed(seed)

    integration_race(linear, [0, 5], expon, 1 - 6*np.exp(-5), True, "Linear function: Exp vs Uni (" + str(seed) + ")")
    integration_race(quadratic, [0, 5], expon, 2 - 37*np.exp(-5), True, "Quadratic function: Exp vs Uni (" + str(seed) + ")")
    integration_race(expon, [0, 5], quadratic, 2 - 37*np.exp(-5), True, "Quadratic function: Qua vs Uni (" + str(seed) + ")")