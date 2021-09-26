import numpy as np
from numpy.random import randint, uniform
from math import pi, cos, sin, sqrt

M = 6
N = 3

L1 = 0.055
L2 = 0.315
L3 = 0.045
L4 = 0.108
L5 = 0.005
L6 = 0.034
L7 = 0.015
L8 = 0.088
L9 = 0.204

x3 = 0
y3 = 0
z3 = 0

POPULATION_SIZE = 100
CHROMOSOME_LENGTH = 10

class InverseKinematicsLGP:

    def __init__(self, M, N, x, y, z):

        self.variable_registers = np.zeros((M, ))
        self.constant_registers = np.zeros((N, ))

        self.x = x
        self.y = y
        self.z = z

        self.x0 = 0
        self.y0 = 0
        self.z0 = 0

        self.theta3 = 0
        self.theta2 = 0
        self.theta1 = 0

        self.population = []
        self.population_fitness = np.zeros(shape=(POPULATION_SIZE, ))

    def init_population(self):
        # Population: a list of lists of tuples, [  [(O, D, O1, O2), (O, D, O1, O2), (O, D, O1, O2)...], 
        #                                           [(O, D, O1, O2), (O, D, O1, O2), (O, D, O1, O2)...]     ]
        for i in range(POPULATION_SIZE):
            current_chromosome = []
            for j in range(CHROMOSOME_LENGTH):
                # Tuple of genes: (Operator, Destination, Operand1, Operand 2)
                current_chromosome.append(( randint(0, 7), 
                                            randint(0, 2), 
                                            randint(0, N+2),
                                            randint(0, N+2)   ))
            self.population.append(current_chromosome)


    def compute_P(self):
        x2 = x3 * cos(self.theta3) - (y3-L9)*sin(self.theta3)
        y2 = x3 * sin(self.theta3) + (y3-L9)*cos(self.theta3)
        z2 = z3

        x1 = (x2+L7)*cos(self.theta2) - (y2-L8)*sin(self.theta2)
        y1 = (x2+L7)*sin(self.theta2) + (y2-L8)*cos(self.theta2)
        z1 = z2 - L5

        self.x0 = (x1+L6)*cos(self.theta1) + (z1+L4)*sin(self.theta1)
        self.y0 = (x1+L6)*sin(self.theta1) - (z1+L4)*cos(self.theta1)
        self.z0 = y1 + L2 + L3

    def randomize_angles(self):
        self.theta1 = uniform(0, 180)
        self.theta2 = uniform(0, 180)
        self.theta3 = uniform(0, 90)

    def compute_error(self):
        euclidian_distance = sqrt((self.x0 - self.x)**2 + (self.y0 - self.y)**2 + (self.z0 - self.z)**2)
        penalty = 0.001 * CHROMOSOME_LENGTH
        return 1/euclidian_distance + penalty

    def compute_fitness(self):
        euclidian_distance = sqrt((self.x0 - self.x)**2 + (self.y0 - self.y)**2 + (self.z0 - self.z)**2)
        return euclidian_distance

if __name__ == "__main__":
    algorithm = InverseKinematicsLGP(3, 3, x=0, y=0, z=0)

    # Assign values to constant registers
    algorithm.constant_registers[0] = pi
    algorithm.constant_registers[1] = 0.5
    algorithm.constant_registers[2] = 0.1

    criteria_met = True
    error = 0

    algorithm.init_population()


    while(not criteria_met):
        algorithm.randomize_angles()
        algorithm.compute_P()
        error = algorithm.compute_error()