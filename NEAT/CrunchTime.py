from Reproduction_2 import crossover, addLayer

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.fitness = 0

def main(numInputs, numOutputs, popSize, maxGens):
    population = []
    for _ in range(popSize):
        individual = NN([numInputs, numOutputs])
        population.append(individual)
    
    
    for _ in range(maxGens):
        offsprings = []
        while len(offsprings) < popSize:
            offspring = NN(crossover(population))
            offspring = NN(addLayer(offspring))
            offsprings.append(offspring)
        


if __name__ == "__main__":
    numInputs = 784
    numOutputs = 10
    popSize = 10
    maxGens = 10
    main()