import numpy as np
from math import pi

M = 6
N = 3

class LGP:

    def __init__(self, M, N):

        self.variable_registers = np.zeros((M, ))
        self.constant_registers = np.zeros((N, ))

        self.population_size = 100
        


if __name__ == "__main__":
    algorithm = LGP(3, 3)

    # Assign values to constant registers
    algorithm.constant_registers[0] = pi
    algorithm.constant_registers[1] = 0.5
    algorithm.constant_registers[2] = 0.1