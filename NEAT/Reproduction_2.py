import random


def addLayer(individual):
    coinflip = random.random()
    if coinflip < 0.5:
        return individual.layers
    else:
        cut = random.randint(1,len(individual.layers)-1)
        newLayer = individual.layers[:cut]
        newLayer.append(random.randint(1,100))
        newLayer.append(individual.layers[cut:])
        return newLayer

def crossover(population):
   parents = random.sample(population,2)
   parent1,parent2 = parents[0].layers,parents[1].layers
   cap = min(len(parent1 ,parent2))
   cut = random.randint(1,cap-1)
   offspring = parent1[:cut]
   offspring.append(parent2[cut:])
   return offspring
