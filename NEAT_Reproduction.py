# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:38:16 2019

@author: NathanLHall
"""

from NEAT_Classes import Genome, ConnectionGene, NodeGene
import random

wMutStep = 2 # weight mutation step size

def addConnectionMutation(self):
    global innovation
    node1 = self.nodes[str(random.choice(self.nodes))]
    node2 = self.nodes[str(random.choice(self.nodes))]
    weight = random.randrange(-2, 2)

    # Avoid connecting nodes in the same input/output layer
    while node1.getType() == node2.getType() and node1.getType() != "HIDDEN":
        node1 = self.nodes[str(random.choice(self.nodes))]
        node2 = self.nodes[str(random.choice(self.nodes))]

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
    for connection in self.connections.values():
        if connection.getInNode() == inNode.getID() and connection.getOutNode() == outNode.getID():
            connectionExists = True
            break

    if connectionExists == False:
        self.addConnectionGene(ConnectionGene(inNode.getID(), outNode.getID(), weight, True, innovation))

def addNodeMutation(self):
    global innovation
    connection = self.connections[str(random.choice(self.connections))]

    inNode = connection.getInNode()
    outNode = connection.getOutNode()

    connection.disable()

    newNode = NodeGene("HIDDEN", len(self.nodes))
    inToNew = ConnectionGene(inNode.getID(), newNode.getID(), 1, True, innovation)
    newToOut = ConnectionGene(newNode.getID(), outNode.getID(), connection.getWeight(), True, innovation)

    self.addNodeGene(newNode)
    self.addConnectionGene(inToNew)
    self.addConnectionGene(newToOut)

def weightMutation(self):
    connection = self.connections[str(random.choice(self.connections))]

    # new = old +/- range(step_size)
    newWeight = connection.getWeight() + ((-1)**random.randint(0, 1)) * random.random(0, wMutStep)

    connection.setWeight(newWeight)

def expressedMutation(self):
    connection = self.connections[str(random.choice(self.connections))]

    if connection.isExpressed():
        connection.disable()
    else:
        connection.enable()

def crossover(parent1, parent2):
    child = Genome()
    fullInheritance = False
    primary = parent1
    secondary = parent2

    if parent1.fitness == parent2.fitness:
        fullInheritance = True
    elif parent2.fitness > parent1.fitness:
        primary = parent2
        secondary = parent1

    for node in primary.getNodeGenes().values():
        child.addNodeGene(node)

    for connection in primary.getConnectionGenes().values():
        if str(connection.innovation) in secondary.getConnectionGenes(): # matching gene
            coinflip = random.randint(0,1)
            if coinflip == 0:
                child.addConnectionGene(connection)
            else:
                child.addConnectionGene(secondary.getConnectionGenes()[str(connection.innovation)])
        else: # disjoint/excess gene
            child.addConnectionGene(connection)

    # If both parents are of equal fitness, inherit all disjoint/excess genes
    if fullInheritance:
        for node in secondary.getNodeGenes().values():
            if str(node.ID) not in child.getNodeGenes():
                child.addNodeGene(node)

        for connection in secondary.getConnectionGenes().values():
            if str(connection.innovation) not in child.getConnectionGenes():
                child.addConnectionGene(connection)

    return child