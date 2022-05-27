from src.library.MC_methods import *
import matplotlib.pyplot as plt
import seaborn as sns

def test_expo_sampling(x_min: float = 0, x_max: float = 5, alpha: float = 1., size: int = 1000000):
    draws = expo_sampling(x_min, x_max, alpha, size)

    sns.displot(draws, bins=np.linspace(x_min - 1, x_max + 1, 50), kde=True)
    plt.show()

def test_numpy_insert():
    array = np.zeros(3)
    index = 1

    array = np.insert(array, index, 1)
    array = np.insert(array, index, 2)

    assert (array == np.array([0., 2., 1., 0., 0.])).all


def test_numpy_delete():
    array = np.array([0, 1, 2, 0, 1])
    index = 1

    array[index - 1] = array[index] + array[index + 1] + array[index - 1]

    array = np.insert(array, index, 1)
    array = np.insert(array, index, 2)

    assert (array == np.array([0., 2., 1., 0., 0.])).all


def test_numpy_prod():
    array = np.array([1,2,3])
    altro = np.array([1, -1, 1])

    assert (array*altro == np.array([1, -2, 3])).all