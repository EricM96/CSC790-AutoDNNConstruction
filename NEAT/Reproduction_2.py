import math
import random

def addLayer(individual):
    coinflip = random.random()
    if len(individual.layers) - 2 == 0:
        adaptiveChance = 1
    else:
        adaptiveChance = 1 / (len(individual.layers) - 2)

    if coinflip < adaptiveChance:
        cut = random.randint(1,len(individual.layers)-1)
        newLayer = individual.layers[:cut]
        newLayer.append(2)
        for item in individual.layers[cut:]:
            newLayer.append(item)
        return newLayer
    else:
        return individual.layers

def addNode(individual):
    if len(individual.layers) - 2 == 0:
        adaptiveChance = 1
    else:
        adaptiveChance = 1 / (len(individual.layers) - 2)

    for i in range(1, len(individual.layers) - 1):
        coinflip = random.random()
        if coinflip < adaptiveChance:
            individual.layers[i] += 1
    return individual.layers

def crossover(population):
   parents = random.sample(population, 2)
   parent1, parent2 = parents[0].layers, parents[1].layers
   cap = min(len(parent1), len(parent2))
   cut = random.randint(1, cap-1)
   offspring = parent1[:cut]
   for item in parent2[cut:]:
       offspring.append(item)
   return offspring


