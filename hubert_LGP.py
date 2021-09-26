import numpy as np
from numpy.random import randint, uniform
from math import pi, cos, sin, sqrt
import matplotlib.pyplot as plt

CHROMOSOME_PENALTY_FACTOR = 0.001

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
CHROMOSOME_LENGTH = 10
TOURNAMENT_SELECTION_PARAMETER = 0.8
CROSSOVER_LIMIT = 3
MUTATION_PROBABILITY = 0.01

NUMBER_OF_GENERATIONS = 1000
NUMBER_OF_POINTS = 20
CROSSOVER_PROBABILITY = 0.7
NUMBER_OF_COPIES = 2

CONSTANT_REGISTERS = [1, 0.001, 0.01, 0.1, 0.5]

class InverseKinematicsLGP:

    def __init__(self, x, y, z):

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
        self.theta1 = uniform(0, 1) * pi
        self.theta2 = uniform(0, 1) * pi
        self.theta3 = uniform(0, 1) * pi / 2

    def compute_fitness(self, chromosome):
        euclidian_distance = sqrt((self.x0 - self.x)**2 + (self.y0 - self.y)**2 + (self.z0 - self.z)**2)
        penalty = CHROMOSOME_PENALTY_FACTOR * len(chromosome)
        return 1/euclidian_distance - penalty

    def decode_chromosome(self, pop_idx):
        registers = np.concatenate(([self.x0, self.y0, self.z0], CONSTANT_REGISTERS))
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

        self.theta1 = registers[0]
        self.theta2 = registers[1]
        self.theta3 = registers[2]

    def tournament_selection(self):

        # Pick random individuals
        individual1 = 0
        individual2 = 0

        while(individual1 == individual2):
            individual1 = randint(0, POPULATION_SIZE - 1)
            individual2 = randint(0, POPULATION_SIZE - 1)
        
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
            random_factor = uniform(0, 1)
            if random_factor < MUTATION_PROBABILITY:
                chromosome[idx] = ( randint(0, 7), 
                                    randint(0, 2),
                                    randint(0, N+2),
                                    randint(0, N+2) )
        return chromosome

    def angles_in_interval(self):
        if ((0 <= self.theta1 <= pi) and 
            (0 <= self.theta2 <= pi) and
            (0 <= self.theta3 <= pi / 2)):
            return True
        else:
            return False

if __name__ == "__main__":
    algorithm = InverseKinematicsLGP(x=0, y=0, z=0)

    all_time_highetst_score = 0
    algorithm.init_population()

    for gen in range(NUMBER_OF_GENERATIONS):

        for points in range(NUMBER_OF_POINTS):
            
            algorithm.randomize_angles()
            algorithm.compute_P()
            for i in range(POPULATION_SIZE):
                #for chromosome, ind, in enumerate(population[i][:]):
                chromosome = algorithm.population[i]
                algorithm.decode_chromosome(i)
                algorithm.compute_P()
                if algorithm.angles_in_interval():
                    algorithm.population_fitness[i] += algorithm.compute_fitness(chromosome)
                else:
                    algorithm.population_fitness[i] -= 100  # ??
        
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
            random_index = randint(0,POPULATION_SIZE-1)
            temp_population[random_index] = fittest_individual

        population = temp_population

        print(len(fittest_individual))
        print(  f"Gen: {gen} Max: {max(algorithm.population_fitness)/NUMBER_OF_POINTS}" + 
                f"Average: {sum(algorithm.population_fitness)/(len(algorithm.population_fitness)*NUMBER_OF_POINTS)}")

        if max(algorithm.population_fitness)/NUMBER_OF_POINTS > 150:
            break
    
    
    plt.figure()
    chromosome = fittest_individual #fittest_individual
    for i in range(4):
        algorithm.randomize_angles()
        algorithm.compute_P()
        coordinate = [algorithm.x0, algorithm.y0, algorithm.z0]
        algorithm.decode_chromosome(fittest_individual_index)
        algorithm.compute_P()
        
        plt.subplot(1,3,1)
        plt.title("X")
        plt.plot(algorithm.x0, coordinate[0], "-o", label=(f"{i}"))
        plt.legend()
        plt.subplot(1,3,2)
        plt.title("Y")
        plt.plot(algorithm.y0, coordinate[1], "-o", label=(f"{i}"))
        plt.legend()
        plt.subplot(1,3,3)
        plt.title("Z")
        plt.plot(algorithm.z0, coordinate[2], "-o", label=(f"{i}"))
        plt.legend()
    plt.show()
