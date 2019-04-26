# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:38:16 2019

@author: NathanLHall
"""

from NEAT_Classes import Genome, ConnectionGene, NodeGene
import random

wMutStep = 2 # weight mutation step size

def addConnection(individual):
    node1 = individual.getRandomNode()
    node2 = individual.getRandomNode()
    weight = random.randrange(-2, 2)

    # Avoid connecting nodes:
    #  - in the same input/output layer
    #  - that are the same
    while (node1.getType() == node2.getType() and node1.getType() != "HIDDEN") or node1.getInnovation() == node2.getInnovation():
        node1 = individual.getRandomNode()
        node2 = individual.getRandomNode()

    # Make sure the connection is flowing from a previous layer to a following layer
    flipped = False # assume the connection is flowing correctly
    if node1.getType() == "HIDDEN" and node2.getType() == "INPUT":
        flipped = True
    elif node1.getType() == "OUTPUT" and node2.getType() == "HIDDEN":
        flipped = True
    elif node1.getType() == "OUTPUT" and node2.getType() == "INPUT":
        flipped = True

    if flipped == False:
        inNode = node1
        outNode = node2
    elif flipped == True:
        inNode = node2
        outNode = node1

    # Check if connection already exists
    connectionExists = False
    for connection in individual.getConnectionGenes().values():
        if connection.getInNode() == inNode.getID() and connection.getOutNode() == outNode.getID():
            connectionExists = True
            break

    if connectionExists == False:
        individual.addConnectionGene(ConnectionGene(inNode.getID(), outNode.getID(), weight, True))

    return individual

def addNode(individual):
    connection = individual.getRandomConnection()

    inNode = connection.getInNode()
    outNode = connection.getOutNode()

    connection.disable()

    newNode = NodeGene("HIDDEN", individual.getNextNodeID())
    inToNew = ConnectionGene(inNode, newNode.getID(), 1, True)
    newToOut = ConnectionGene(newNode.getID(), outNode, connection.getWeight(), True)

    individual.addNodeGene(newNode)
    individual.addConnectionGene(inToNew)
    individual.addConnectionGene(newToOut)

    return individual

def mutateWeights(individual):
    for connection in individual.getConnectionGenes().values():
        # 90% chance of modifying current weight within a step size
        # 10% chance of replacing current weight with a new random value
        lottery = random.uniform(0, 1)
        if lottery <= 0.9:
            # new = old + k, where -2 <= k <= 2
            newWeight = connection.getWeight() + random.uniform(-wMutStep, wMutStep)
        else:
            newWeight = random.uniform(-2, 2)

        connection.setWeight(newWeight)

    return individual

def expressedMutation(individual):
    connection = individual.getRandomConnection()

    # Make sure turning this connection off won't fragment the network
    safeguard = checkNumConnections(individual, connection)
    count = 0 # If there are no possible connections to turn off, kill after a certain count
    while safeguard == "Unsafe":
        if count >= 10:
            return individual
        connection = individual.getRandomConnection()
        safeguard = checkNumConnections(individual, connection)
        count += 1

    if connection.isExpressed():
        connection.disable()
    else:
        connection.enable()

    return individual

def checkNumConnections(individual, connection):
    inNode = connection.getInNode()
    outNode = connection.getOutNode()

    # Check number of outgoing connections from inNode
    inCount = 0
    for key in individual.connections.keys():
        if individual.connections[key].inNode == inNode and individual.connections[key].expressed == True:
            inCount += 1

    # Check number of incoming connections from outNode
    outCount = 0
    for key in individual.connections.keys():
        if individual.connections[key].outNode == outNode and individual.connections[key].expressed == True:
            outCount += 1

    if inCount > 1 and outCount > 1:
        return "Safe"
    else:
        return "Unsafe"

def crossover(parent1, parent2):
    child = Genome()
    fullInheritance = False
    primary = parent1
    secondary = parent2

    if parent1.getFitness() == parent2.getFitness():
        fullInheritance = True
    elif parent2.getFitness() > parent1.getFitness():
        primary = parent2
        secondary = parent1

    # Inherit all nodes from primary parent
    for node in primary.getNodeGenes().values():
        child.addNodeGene(node)

    # Inherit connections
    for connection in primary.connections.values():
        if str(connection.getInnovation()) in secondary.getConnectionGenes().keys(): # matching gene
            coinflip = random.randint(0,1) # inherit from random parent
            if coinflip == 0:
                child.addConnectionGene(connection)
            else:
                child.addConnectionGene(secondary.getConnectionGenes()[str(connection.getInnovation())])
        else: # disjoint/excess gene
            child.addConnectionGene(connection) # inherit all disjoint/excess genes from primary parent

    # If both parents are of equal fitness, inherit all disjoint/excess genes
    if fullInheritance:
        for node in secondary.getNodeGenes().values():
            if str(node.getInnovation()) not in child.getNodeGenes().keys():
                child.addNodeGene(node)

        for connection in secondary.getConnectionGenes().values():
            if str(connection.getInnovation()) not in child.getConnectionGenes().keys():
                child.addConnectionGene(connection)

    return child