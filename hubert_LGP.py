from typing import ForwardRef
import numpy as np
from numpy.random import randint, uniform
from math import pi, cos, sin, sqrt
import matplotlib.pyplot as plt

CHROMOSOME_PENALTY_FACTOR = 0.000001

M = 6
N = 5

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
CHROMOSOME_LENGTH = 50
TOURNAMENT_SELECTION_PARAMETER = 0.8
CROSSOVER_LIMIT = 3
MUTATION_PROBABILITY = 0.001

NUMBER_OF_GENERATIONS = 100
NUMBER_OF_POINTS = 25
CROSSOVER_PROBABILITY = 0.7
NUMBER_OF_COPIES = 2

CONSTANT_REGISTERS = [1, 0.001, 0.01, 0.1, 0.5]

class InverseKinematicsLGP:

    def __init__(self):

        self.x = 0
        self.y = 0
        self.z = 0

        self.x0 = np.zeros((NUMBER_OF_POINTS, ))
        self.y0 = np.zeros((NUMBER_OF_POINTS, ))
        self.z0 = np.zeros((NUMBER_OF_POINTS, ))

        self.theta3 = np.zeros((NUMBER_OF_POINTS, ))
        self.theta2 = np.zeros((NUMBER_OF_POINTS, ))
        self.theta1 = np.zeros((NUMBER_OF_POINTS, ))

        self.population = []
        self.population_fitness = np.zeros(shape=(POPULATION_SIZE, ))

        self.euclidian_error = 0

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


    def compute_P(self, current_P):
        x2 = x3 * cos(self.theta3[current_P]) - (y3-L9)*sin(self.theta3[current_P])
        y2 = x3 * sin(self.theta3[current_P]) + (y3-L9)*cos(self.theta3[current_P])
        z2 = z3

        x1 = (x2+L7)*cos(self.theta2[current_P]) - (y2-L8)*sin(self.theta2[current_P])
        y1 = (x2+L7)*sin(self.theta2[current_P]) + (y2-L8)*cos(self.theta2[current_P])
        z1 = z2 - L5

        self.x0[current_P] = (x1+L6)*cos(self.theta1[current_P]) + (z1+L4)*sin(self.theta1[current_P])
        self.y0[current_P] = (x1+L6)*sin(self.theta1[current_P]) - (z1+L4)*cos(self.theta1[current_P])
        self.z0[current_P] = y1 + L2 + L3

    def forward(self, current_P):
        x2 = x3 * cos(self.theta3[current_P]) - (y3-L9)*sin(self.theta3[current_P])
        y2 = x3 * sin(self.theta3[current_P]) + (y3-L9)*cos(self.theta3[current_P])
        z2 = z3

        x1 = (x2+L7)*cos(self.theta2[current_P]) - (y2-L8)*sin(self.theta2[current_P])
        y1 = (x2+L7)*sin(self.theta2[current_P]) + (y2-L8)*cos(self.theta2[current_P])
        z1 = z2 - L5

        self.x = (x1+L6)*cos(self.theta1[current_P]) + (z1+L4)*sin(self.theta1[current_P])
        self.y = (x1+L6)*sin(self.theta1[current_P]) - (z1+L4)*cos(self.theta1[current_P])
        self.z = y1 + L2 + L3

    def randomize_angles(self, current_P):
        self.theta1[current_P] = uniform(0, 1) * pi
        self.theta2[current_P] = uniform(0, 1) * pi
        self.theta3[current_P] = uniform(0, 1) * pi / 2

    def compute_fitness(self, chromosome, current_P):
        euclidian_distance = sqrt(  (self.x0[current_P] - self.x)**2 + 
                                    (self.y0[current_P] - self.y)**2 + 
                                    (self.z0[current_P] - self.z)**2    )
        penalty = CHROMOSOME_PENALTY_FACTOR * len(chromosome)    
        self.euclidian_error += euclidian_distance
        return 1/euclidian_distance - penalty

    def decode_chromosome(self, pop_idx, current_P):
        registers = np.concatenate(([   self.x0[current_P], 
                                        self.y0[current_P], 
                                        self.z0[current_P]], 
                                        CONSTANT_REGISTERS) )
        skip_instruction = False

        chromosome = self.population[pop_idx]
        
        for gene in chromosome:
            if skip_instruction:
                # PASS
                skip_instruction = False
            else:
                (operator, destination, operand1, operand2) = gene
                if operator == 0: # Operation: +
                    registers[destination] = operand1 + operand2
                elif operator == 1: # Operation: -
                    registers[destination] = operand1 - operand2
                elif operator == 2: # Operation: *
                    registers[destination] = operand1 * operand2
                elif operator == 3: # Operation: /
                    if not operand2 == 0: 
                        registers[destination] = operand1 / operand2
                elif operator == 4: # Operation: cos
                    registers[destination] = cos(operand1)
                elif operator == 5: # Operation: sin
                    registers[destination] = sin(operand2)
                elif operator == 6: # Operation: <=
                    if operand1 <= operand2:
                        skip_instruction = True
                elif operator == 7: # Operation: >
                    if operand1 > operand2:
                        skip_instruction = True

        self.theta1[current_P] = registers[0]
        self.theta2[current_P] = registers[1]
        self.theta3[current_P] = registers[2]

    def tournament_selection(self):

        # Pick random individuals
        individual1 = 0
        individual2 = 0

        while(individual1 == individual2):
            individual1 = randint(0, POPULATION_SIZE)
            individual2 = randint(0, POPULATION_SIZE)
        
        random_factor = uniform(0.0, 1.0)

        HIGHEST_FITNESS_WINS = random_factor < TOURNAMENT_SELECTION_PARAMETER

        if self.population_fitness[individual1] >= self.population_fitness[individual2]:
            if HIGHEST_FITNESS_WINS:
                return individual1
            else:
                return individual2
        else:
            if HIGHEST_FITNESS_WINS:
                return individual2
            else:
                return individual1

    def chromosome_crossover(self, pop_idx1, pop_idx2):

        chromosome1 = self.population[pop_idx1]
        chromosome2 = self.population[pop_idx2]

        if len(chromosome1) >= CROSSOVER_LIMIT and len(chromosome2) >= CROSSOVER_LIMIT:

            # TEST THIS FUNCTION IN SEPARATE FILE

            chromo1_split_idx1 = randint(0, len(chromosome1))
            chromo1_split_idx2 = randint(0, len(chromosome1))

            chromo2_split_idx1 = randint(0, len(chromosome2))
            chromo2_split_idx2 = randint(0, len(chromosome2))

            chromo1_split1  = chromosome1[0:chromo1_split_idx1]
            chromo1_split2 = chromosome1[chromo1_split_idx1:chromo1_split_idx2]
            chromo1_split3 = chromosome1[chromo1_split_idx2:len(chromosome1)]

            chromo2_split1  = chromosome2[0:chromo2_split_idx1]
            chromo2_split2 = chromosome2[chromo2_split_idx1:chromo2_split_idx2]
            chromo2_split3  = chromosome2[chromo2_split_idx2:len(chromosome2)]

            crossed_chromosome1 = chromo1_split1 + chromo2_split2 + chromo1_split3
            crossed_chromosome2 = chromo2_split1 + chromo1_split2 + chromo2_split3

            self.population[pop_idx1] = crossed_chromosome1
            self.population[pop_idx2] = crossed_chromosome2

    def random_mutation(self, chromosome):
        for idx, _ in enumerate(chromosome):
            for gene in range(4):
                instruction = np.asarray(chromosome[idx])
                random_factor = uniform(0, 1)
                if random_factor < MUTATION_PROBABILITY:
                    if gene == 0:
                        instruction[gene] = randint(0, 7)
                    elif gene == 1:
                        instruction[gene] = randint(0, 2)
                    else:
                        instruction[gene] = randint(0, N+2)
            chromosome[idx] = ( instruction[0],
                                instruction[1],
                                instruction[2],
                                instruction[3]  )
        return chromosome

    def angles_in_interval(self, current_p):
        if ((0 <= self.theta1[current_p] <= pi) and 
            (0 <= self.theta2[current_p] <= pi) and
            (0 <= self.theta3[current_p] <= pi / 2)):
            return True
        else:
            return False

if __name__ == "__main__":
    algorithm = InverseKinematicsLGP()

    all_time_highetst_score = 0
    algorithm.init_population()

    for current_P in range(NUMBER_OF_POINTS):
            
            algorithm.randomize_angles(current_P)
            algorithm.compute_P(current_P)

    for gen in range(NUMBER_OF_GENERATIONS):

        algorithm.population_fitness = np.zeros(shape=(POPULATION_SIZE, ))

        for i in range(POPULATION_SIZE):
            chromosome = algorithm.population[i]
            for current_P in range(NUMBER_OF_POINTS):
                algorithm.decode_chromosome(i, current_P)
                algorithm.forward(current_P)
                if algorithm.angles_in_interval(current_P):
                    algorithm.population_fitness[i] += algorithm.compute_fitness(chromosome, current_P)
                    if len(algorithm.population[i]) < 10:
                        algorithm.population_fitness[i] -= 10
                else:
                    algorithm.population_fitness[i] -= 10  # Maybe not so high penalty?

        temp_population = []
        # Save fittest individual
        fittest_individual_index = np.argmax(algorithm.population_fitness)
        fittest_individual = algorithm.population[fittest_individual_index]
        fitness_score_best_ind = algorithm.population_fitness[fittest_individual_index]
        if fitness_score_best_ind > all_time_highetst_score:
            all_time_fittest = fittest_individual
            all_time_highetst_score = fitness_score_best_ind
        # Tournament_selection & Crossover
        for i in range(0, POPULATION_SIZE, 2):
            # tournament selction
            i1 = algorithm.tournament_selection()
            i2 = algorithm.tournament_selection()

            # crossover
            r = uniform(0, 1)
            if r < CROSSOVER_PROBABILITY:
                algorithm.chromosome_crossover(i1, i2)

            temp_population.append(algorithm.population[i1])
            temp_population.append(algorithm.population[i2])
        

        # mutation
        for i in range(POPULATION_SIZE):
            original_chromosome = temp_population[i]
            mutated_chromosome = algorithm.random_mutation(original_chromosome)
            temp_population[i] = mutated_chromosome
        # insert fittest individual (elitism) 
        for i in range(NUMBER_OF_COPIES):
            random_index = randint(0,POPULATION_SIZE)
            temp_population[random_index] = all_time_fittest

        algorithm.population = np.copy(temp_population)

        print(  f"\nGen: {gen} Max: {max(algorithm.population_fitness)/NUMBER_OF_POINTS}" + 
                f"\tAverage: {sum(algorithm.population_fitness)/(len(algorithm.population_fitness)*NUMBER_OF_POINTS)}")
        print("\nAVG ERROR: " + str(algorithm.euclidian_error / (NUMBER_OF_POINTS * POPULATION_SIZE)))
        print("ALL TIME FITTEST: " + str(all_time_highetst_score/NUMBER_OF_POINTS))
        print("CHROMOSOME LENGTH: " + str(len(all_time_fittest)))

        algorithm.euclidian_error = 0
