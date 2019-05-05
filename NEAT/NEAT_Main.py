# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:12:06 2019

@author: NathanLHall
"""

import NEAT_Classes
import NEAT_Reproduction
import NEAT_Speciation
from utilities import compute_fitness

# import numpy as np
import random

def initializePop(numInputs, numOutputs, popSize):
    inputs = [NEAT_Classes.NodeGene("INPUT", n) for n in range(1, numInputs + 1)]
    outputs = [NEAT_Classes.NodeGene("OUTPUT", n) for n in range(numInputs + 1, numInputs + numOutputs + 1)]

    connections = []
    for inNode in inputs:
        for outNode in outputs:
            newConnection = NEAT_Classes.ConnectionGene(inNode.ID, outNode.ID, 1, True)
            connections.append(newConnection)
            NEAT_Reproduction.recordConnection(newConnection)

    population = []
    for _ in range(popSize):
        individual = NEAT_Classes.Genome()

        for node in inputs:
            individual.addNodeGene(node)
        for node in outputs:
            individual.addNodeGene(node)
        for connection in connections:
            individual.addConnectionGene(connection)

        population.append(individual)
        
    return population

def main(numInputs, numOutputs, popSize, maxGenerations, distanceThreshold):
    population = initializePop(numInputs, numOutputs, popSize)
    for individual in population:
        individual.fitness = compute_fitness(individual)

    species = NEAT_Speciation.species

    newSpecies = NEAT_Classes.Species(0)
    newSpecies.update(population[0], population)
    species.append(newSpecies)

    for generation in range(1, maxGenerations):
        print("------------------------------")
        print("Generation:", generation)
        offsprings = []
        for parent1 in population:
            parent2 = random.choice(population)

            # Avoid crossover using the same parent with itself
            while parent1 == parent2:
                parent2 = random.choice(population)
            offspring = NEAT_Reproduction.crossover(parent1, parent2)
            offspring = NEAT_Reproduction.addConnection(offspring)
            offspring = NEAT_Reproduction.addNode(offspring)
            # offspring = NEAT_Reproduction.mutateWeights(offspring)
            offspring = NEAT_Reproduction.expressedMutation(offspring)
            offsprings.append(offspring)

        offsprings = NEAT_Reproduction.cleanConnections(offsprings)

        # Measure offsprings fitnesses
        for offspring in offsprings:
            offspring.fitness = compute_fitness(offspring)

        nextGeneration = []
        while len(nextGeneration) < popSize:
            individual = random.choice(population)
            offspring = random.choice(offsprings)
            if offspring.fitness >= individual.fitness:
                nextGeneration.append(offspring)
            else:
                nextGeneration.append(individual)
        
        population = nextGeneration

        # species = NEAT_Speciation.assignSpecies(offsprings, generation, distanceThreshold, species, popSize)
        # species = NEAT_Speciation.cullSpecies(species, popSize)
        # species = NEAT_Speciation.updateRepresentative(species, generation)

        # population = []
        # for s in species:
            # for member in s.members.values():
            #     population.append(member)
        
if __name__ == "__main__":
    numInputs = 3
    numOutputs = 1
    popSize = 50
    maxGens = 30
    distanceThreshold = 3.0
    main(numInputs, numOutputs, popSize, maxGens, distanceThreshold)