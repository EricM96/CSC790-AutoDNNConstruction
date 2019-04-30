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

    nodes1 = genome1.getNodeGenes()
    nodes2 = genome2.getNodeGenes()
    maxNodeID1 = max(node.ID for node in nodes1)
    maxNodeID2 = max(node.ID for node in nodes2)
    maxNodeID = max([maxNodeID1, maxNodeID2])

    nodeIDs = []
    for node in nodes1:
        if node.ID not in nodeIDs:
            nodeIDs.append(node.ID)
    for node in nodes2:
        if node.ID not in nodeIDs:
            nodeIDs.append(node.ID)

    for i in nodeIDs:
        node1 = None
        node2 = None
        
        for node in nodes1:
            if i == node.ID:
                node1 = node
        for node in nodes2:
            if i == node.ID:
                node2 = node

        if node1 != None and node2 != None:
            matchingGenes += 1

        elif node1 == None and maxNodeID1 < i and node2 != None:
            excessGenes += 1
        elif node1 != None and maxNodeID2 < i and node2 == None:
            excessGenes += 1

        elif node1 == None and maxNodeID1 > i and node2 != None:
            disjointGenes += 1
        elif node1 != None and maxNodeID2 > i and node2 == None:
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
def speciate(population, generation, distanceThreshold, species):
    newMembers = []
    for individual in population:
        candidates = []
        assigned = False
        for s in species:
            delta = compatibilityDistance(individual, s.representative)
            if  delta < distanceThreshold:
                candidates.append((delta, s.ID))
                assigned = True

        if assigned == False:
            newSpecies = NEAT_Classes.Species(generation)
            newSpecies.update(individual, [individual])
            species.append(newSpecies)
            continue

        candidates.sort(key=lambda tup: tup[0])
        closestSpecies = candidates[0][1]
        newMembers.append((closestSpecies, individual))

    for memberTuple in newMembers:
        # Update the individual's species ID
        memberTuple[1].species = memberTuple[0]
        # Add individual to species members
        species[memberTuple[0]].members[str(memberTuple[1].ID)] = copy.deepcopy(memberTuple[1])

    return species

def cullSpecies(species, maxPopSize):
    totalMembers = 0
    for s in species:
        totalMembers += len(s.members)

    while totalMembers > maxPopSize:
        largest = 0
        index = 0
        for i in range(len(species)):
            if len(species[i].members) > largest:
                largest = len(species[i].members)
                index = i

        neighborMeasures = []
        for member1 in species[index].members.values():
            proximities = 0

            for member2 in species[index].members.values():
                proximities += compatibilityDistance(member1, member2)
            
            neighborMeasures.append((proximities, member1))

        neighborMeasures.sort(key=lambda tup: tup[0], reverse=True)

        furthestMemberID = neighborMeasures[0][1].ID
        species[index].members.pop(str(furthestMemberID))

        totalMembers = 0
        for s in species:
            totalMembers += len(s.members)

    return species