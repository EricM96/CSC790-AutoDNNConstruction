# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 00:44:50 2019

@author: NathanLHall
"""

c1 = 1.0
c2 = 1.0
c3 = 0.4
distanceThreshold = 3.0
species = []

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
def speciate(population):
    unspeciated = set(population)
#    newRepresentatives = {}
#    newMembers = {}
    for individual in unspeciated:
        candidates = []
        for s in species:
            delta = compatibilityDistance(individual, s.representative)
            if  delta < distanceThreshold:
                candidates.append((delta, individual))

            dummy, newRepresentative = min(candidates, key=lambda x: x[0])
            newRepresentatives[str(s.ID)] = newRepresentative
            newMembers[str(s.ID)] = [newRepresentative]