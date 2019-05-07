import copy
import random
from fitness2 import main
from Reproduction_2 import crossover, addLayer, addNode
from utilities import compute_fitness

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.fitness = 0

def main(numInputs, numOutputs, popSize, maxGens):
    population = []
    for _ in range(popSize):
        individual = NN([numInputs, numOutputs])
        population.append(individual)
    
    initialFitness == compute_fitness(population[0])
    for individual in population:
        individual.fitness = initialFitness
    
    bestSolution = copy.deepcopy(population[0])
    
    for _ in range(maxGens):
        offsprings = []
        while len(offsprings) < popSize:
            offspring = NN(crossover(population))
            offspring = NN(addLayer(offspring))
            offspring.fitness = compute_fitness(offspring)
            offsprings.append(offspring)
        
        for offspring in offsprings:
            if offspring.fitness >= bestSolution.fitness:
                bestSolution = copy.deepcopy(offspring)
            

        nextGen = []
        while len(nextGen) < popSize:
            parent = random.choice(population)
            offspring = random.choice(offsprings)
            if offspring.fitness >= parent.fitness:
                nextGen.append(offspring)
            else:
                nextGen.append(parent)
        
        population = nextGen

    return bestSolution


if __name__ == "__main__":
    numInputs = 784
    numOutputs = 10
    popSize = 10
    maxGens = 10
    result = main(numInputs, numOutputs, popSize, maxGens)
    print("Fitness", result.fitness)
    print("NN Structure", result.layers)