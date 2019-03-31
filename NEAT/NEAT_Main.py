# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:12:06 2019

@author: NathanLHall
"""

import NEAT_Classes
import NEAT_Reproduction
import NEAT_Speciation

import numpy as np
import random

numInputs = 3
numOutputs = 1
popSize = 3
maxGenerations = 5

def initializePop(numInputs, numOutputs, popSize):
    inputs = [NEAT_Classes.NodeGene("INPUT", n) for n in range(1, numInputs + 1)]
    outputs = [NEAT_Classes.NodeGene("OUTPUT", n) for n in range(numInputs + 1, numInputs + numOutputs + 1)]

    connections = []
    for inNode in inputs:
        for outNode in outputs:
            connections.append(NEAT_Classes.ConnectionGene(inNode.getID(), outNode.getID(), 1, True))

    population = []
    for _ in range(popSize):
        individual = NEAT_Classes.Genome()
        individual.ID = 0

        for node in inputs:
            individual.addNodeGene(node)
        for node in outputs:
            individual.addNodeGene(node)
        for connection in connections:
            individual.addConnectionGene(connection)

        population.append(individual)

    A = NEAT_Classes.Species(0)
    A.ID = 0
    A.update(population[0], [population[0]])
    NEAT_Speciation.species.append(A)

    NEAT_Classes.genomeID = 1
    NEAT_Classes.innovationNod = 0
    NEAT_Classes.innovationCon = 0
    NEAT_Classes.speciesID = 0

    return population, NEAT_Speciation.species

def main():
    population, species = initializePop(numInputs, numOutputs, popSize)

    generation = 1
    for _ in range(maxGenerations):
        for specie in species:
            for member

#        parents = random.sample(population, 2)

#    NEAT_Reproduction.mutateWeights()

    # 80% chance of having a weight mutated
#    lottery = np.random.uniform(0, 1)
#    if lottery > 0.8:
#        continue

main()