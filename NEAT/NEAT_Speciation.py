# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 00:44:50 2019

@author: NathanLHall
"""

import NEAT_Classes

import copy

c1 = 1.0
c2 = 1.0
c3 = 0.4
species = []
maxNumSpeciesMembers = 10

def compatibilityDistance(genome1, genome2):
    excessGenes, disjointGenes, avgWeightDifference = compareGenes(genome1, genome2)
    nodeCount1 = len(genome1.getNodeGenes())
    nodeCount2 = len(genome2.getNodeGenes())
    connectionCount1 = len(genome1.getConnectionGenes())
    connectionCount2 = len(genome2.getConnectionGenes())

    geneCount1 = nodeCount1 + connectionCount1
    geneCount2 = nodeCount2 + connectionCount2

    if geneCount1 < 20 and geneCount2 < 20:
        N = 1
    else:
        N = max(geneCount1, geneCount2)

    delta = (c1 * excessGenes + c2 * disjointGenes) / N + c3 * avgWeightDifference

    return delta

def compareGenes(genome1, genome2):
    matchingGenes = 0
    excessGenes = 0
    disjointGenes = 0
    weightDifference = 0.0

    maxNodeInnov1 = int(max(genome1.getNodeGenes().keys()))
    maxNodeInnov2 = int(max(genome2.getNodeGenes().keys()))

    if maxNodeInnov1 >= maxNodeInnov2:
        maxNodeInnov = maxNodeInnov1
    else:
        maxNodeInnov = maxNodeInnov2

    for i in range(0, maxNodeInnov + 1):
        node1 = None
        node2 = None

        if str(i) in genome1.getNodeGenes().keys():
            node1 = genome1.getNodeGenes()[str(i)]
        if str(i) in genome2.getNodeGenes().keys():
            node2 = genome2.getNodeGenes()[str(i)]

        if node1 != None and node2 != None:
            matchingGenes += 1

        elif node1 == None and maxNodeInnov1 < i and node2 != None:
            excessGenes += 1
        elif node1 != None and maxNodeInnov2 < i and node2 == None:
            excessGenes += 1

        elif node1 == None and maxNodeInnov1 > i and node2 != None:
            disjointGenes += 1
        elif node1 != None and maxNodeInnov2 > i and node2 == None:
            disjointGenes += 1

    maxConnectionInnov1 = int(max(genome1.getConnectionGenes().keys()))
    maxConnectionInnov2 = int(max(genome2.getConnectionGenes().keys()))

    if maxConnectionInnov1 >= maxConnectionInnov2:
        maxConnectionInnov = maxConnectionInnov1
    else:
        maxConnectionInnov = maxConnectionInnov2

    for i in range(0, maxConnectionInnov + 1):
        connection1 = None
        connection2 = None

        if str(i) in genome1.getConnectionGenes().keys():
            connection1 = genome1.getConnectionGenes()[str(i)]
        if str(i) in genome2.getConnectionGenes().keys():
            connection2 = genome2.getConnectionGenes()[str(i)]

        if connection1 != None and connection2 != None:
            matchingGenes += 1
            weightDifference += abs(connection1.getWeight() - connection2.getWeight())

        elif connection1 == None and maxConnectionInnov1 < i and connection2 != None:
            excessGenes += 1
        elif connection1 != None and maxConnectionInnov2 < i and connection2 == None:
            excessGenes += 1

        elif connection1 == None and maxConnectionInnov1 > i and connection2 != None:
            disjointGenes += 1
        elif connection1 != None and maxConnectionInnov2 > i and connection2 == None:
            disjointGenes += 1

    return excessGenes, disjointGenes, weightDifference / matchingGenes

# TO-DO
# Take in a population of unidentified species, and assign them to the proper species
def speciate(population, generation, distanceThreshold):
    newMembers = []
    for individual in population:
        candidates = []
        assigned = False
        for s in species:
            delta = compatibilityDistance(individual, s.representative)
            if  delta < distanceThreshold:
                candidates.append((delta, individual, s.ID))
                assigned = True

        if assigned == False:
            newSpecies = NEAT_Classes.Species(generation)
            newSpecies.update(individual, [individual])
            species.append(newSpecies)
        else:
            candidates.sort(key=lambda tup: tup[0])
            closestSpecies = candidates[0][2]
            newMembers.append((closestSpecies, candidates[0][1]))

    for memberTuple in newMembers:
        species[memberTuple[0]].members[str(copy.deepcopy(memberTuple[1].ID))] = copy.deepcopy(memberTuple[1])

    return species

def cullSpecies(self):
    for s in species:
        for member1 in s.members:
            proximities = 0
            neighborMeasures = []

            for member2 in s.members:
                proximities += compatibilityDistance(member1, member2)
            
            neighborMeasures.append((proximities, member1))
            neighborMeasures.sort(key=lambda tup: tup[0])

            #for i in range(maxNumSpeciesMembers)