import random


def addLayer(individual):
    coinflip = random.random()
    if coinflip < 0.5:
        return individual
    else:
        cut = random.randint(1,len(individual)-1)
        newLayer = individual[:cut]
        newLayer.append(random.randint(1,100))
        newLayer.append(individual[cut:])
        return newLayer

def crossover(population):
   parents = random.sample(population,2)
   parent1,parent2 = parents[0],parents[1]
   cap = min(len(parent1 ,parent2))
   cut = random.randint(1,cap-1)
   offspring = parent1[:cut]
   offspring.append(parent2[cut:]
   return offspring


