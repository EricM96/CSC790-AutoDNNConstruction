import random

def addLayer(individual):
    coinflip = random.random()
    if len(individual.layers) - 2 == 0:
        adaptiveChance = 1:
    else:
        adaptiveChance = 1 / (len(individual.layers) - 2)

    if coinflip < adaptiveChance:
        cut = random.randint(1,len(individual.layers)-1)
        newLayer = individual.layers[:cut]
        newLayer.append(random.randint(1,100))
        newLayer.append(individual.layers[cut:])
        return newLayer
    else:
        return individual.layers

def addNode(individual):
    if len(individual.layers) - 2 == 0:
        adaptiveChance = 1:
    else:
        adaptiveChance = 1 / (len(individual.layers) - 2)

    for i in range(1, len(individual.layers) - 2):
        coinflip = random.random()
        if coinflip < adaptiveChance:
            individual.layers[i] += 1

def crossover(population):
   parents = random.sample(population, 2)
   parent1,parent2 = parents[0].layers, parents[1].layers
   cap = min(len(parent1 , parent2))
   cut = random.randint(1, cap-1)
   offspring = parent1[:cut]
   offspring.append(parent2[cut:])
   return offspring


