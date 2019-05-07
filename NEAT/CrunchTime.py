import copy
import random
from fitness2 import main as fitMain
from Reproduction_2 import crossover, addLayer, addNode
from utilities import compute_fitness

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.fitness = 0
        self.generation = None

bestSolutions = []
def main(numInputs, numOutputs, popSize, maxGens):
    population = []
    for _ in range(popSize):
        individual = NN([numInputs, 1, numOutputs])
        individual.generation = 0
        population.append(individual)
    
    initialFitness = fitMain(population[0].layers)
    for individual in population:
        individual.fitness = initialFitness
    
    bestSolutions.append(copy.deepcopy(population[0]))
    
    i = 1
    while True:
    # for i in range(1, maxGens + 1):
        offsprings = []
        while len(offsprings) < popSize:
            offspring = NN(crossover(population))
            offspring.layers = addLayer(offspring)
            offspring.layers = addNode(offspring)
            offspring.fitness = fitMain(offspring.layers)
            offspring.generation = i
            offsprings.append(offspring)

        for offspring in offsprings:
            if offspring.fitness >= bestSolutions[-1].fitness:
                bestSolutions.append(copy.deepcopy(offspring))
            
        pool = []
        for j in range(popSize):
            pool.append(population[j])
            pool.append(offsprings[j])

        nextGen = []
        while len(nextGen) < popSize:
            ind1, ind2 = random.sample(pool, 2)
            if ind1.fitness >= ind2.fitness:
                nextGen.append(ind1)
            else:
                nextGen.append(ind2)
        
        print("Generation", i, "complete:")
        print("    Best fitness    -", bestSolutions[-1].fitness)
        print("    Best Topology   -", bestSolutions[-1].layers)
        print("    From Generation -", bestSolutions[-1].generation)
        print()
        with open(PATH, 'w', newline='\n') as file:
            for champion in bestSolutions:
                file.write(str(champion.fitness) + ' ' + str(champion.layers) + ' ' + str(champion.generation) + '\n')

        population = nextGen
        i += 1

    return bestSolutions[-1]


if __name__ == "__main__":
    PATH = "D:\DL-EvoNN_fitness_report.txt"
    numInputs = 784
    numOutputs = 10
    popSize = 5
    maxGens = 10
    result = main(numInputs, numOutputs, popSize, maxGens)
    print("Fitness", result.fitness)
    print("NN Structure", result.layers)