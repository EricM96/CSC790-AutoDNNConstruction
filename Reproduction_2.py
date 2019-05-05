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
