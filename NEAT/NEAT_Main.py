# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:12:06 2019

@author: NathanLHall
"""

import NEAT_Classes
import NEAT_Reproduction
import NEAT_Speciation

# import numpy as np
import random

def initializePop(numInputs, numOutputs, popSize):
    inputs = [NEAT_Classes.NodeGene("INPUT", n) for n in range(1, numInputs + 1)]
    outputs = [NEAT_Classes.NodeGene("OUTPUT", n) for n in range(numInputs + 1, numInputs + numOutputs + 1)]

    connections = []
    for inNode in inputs:
        for outNode in outputs:
            connections.append(NEAT_Classes.ConnectionGene(inNode.ID, outNode.ID, 1, True))

    population = []
    for _ in range(popSize):
        individual = NEAT_Classes.Genome()
        individual.species = 0

        for node in inputs:
            individual.addNodeGene(node)
        for node in outputs:
            individual.addNodeGene(node)
        for connection in connections:
            individual.addConnectionGene(connection)

        population.append(individual)
        
    generation = 0
    newSpecies = NEAT_Classes.Species(generation)
    newSpecies.update(population[0], population)
    
    species = NEAT_Speciation.species
    species.append(newSpecies)
    # NEAT_Classes.genomeID = 1
    # NEAT_Classes.innovationNod = 0
    # NEAT_Classes.innovationCon = 0
    # NEAT_Classes.speciesID = 0

    return population, species

def main(numInputs, numOutputs, popSize, maxGenerations, distanceThreshold):
    population, species = initializePop(numInputs, numOutputs, popSize)
    # ? Measure fitness
    for generation in range(1, maxGenerations):
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

        # Measure offsprings fitnesses
        species = NEAT_Speciation.speciate(offsprings, generation, distanceThreshold, species)
        species = NEAT_Speciation.cullSpecies(species, popSize)
    #     print("Generation:", generation)
    # print(len(species))
        # Update species representative (most fit? center most?)
        # Cull species members
        # Measure species fitnesses


    # for offspring in offsprings:
    #     offspring.displayConnectionGenes()
    #print(len(offsprings))

if __name__ == "__main__":
    numInputs = 3
    numOutputs = 1
    popSize = 10
    maxGens = 10
    distanceThreshold = 3.0
    main(numInputs, numOutputs, popSize, maxGens, distanceThreshold)