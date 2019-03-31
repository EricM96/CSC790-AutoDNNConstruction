# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:51:14 2019

@author: NathanLHall
"""

import copy
import random

genomeID = 1
innovationNod = 0
innovationCon = 0
speciesID = 0

# Genome = Node Genes + Connections Genes
class Genome:
    def __init__(self):
        global genomeID
        self.ID = genomeID
        self.nodes = {} # {"key" : "value"} = {"node.innovation" : node}
        self.connections = {} # {"key" : "value"} = {"connection.innovation" : connection}
        self.fitness = None
        self.species = None
        genomeID += 1

    def addNodeGene(self, node):
        self.nodes[str(copy.deepcopy(node.getInnovation()))] = copy.deepcopy(node)

    def getNodeGenes(self):
        return self.nodes

    def getRandomNode(self):
        key = str(random.sample(self.getNodeGenes().keys(), 1)[0])
        return self.getNodeGenes()[key]

    def getNextNodeID(self):
        maxID = 0
        for node in self.getNodeGenes().values():
            if node.getID() > maxID:
                maxID = node.getID()
        return maxID + 1

    def addConnectionGene(self, connection):
        self.connections[str(copy.deepcopy(connection.getInnovation()))] = copy.deepcopy(connection)

    def getConnectionGenes(self):
        return self.connections

    def getRandomConnection(self):
        key = str(random.sample(self.getConnectionGenes().keys(), 1)[0])
        return self.getConnectionGenes()[key]

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

    def displayConnectionGenes(self):
        print("--------------------------------------------------")
        print("                  NETWORK GENOME                  ")
        print("FITNESS:", self.getFitness())
        print()
        print("NODE GENES:")
        for innovNum in sorted(self.getNodeGenes().keys(), key=lambda s: int(s)):
            node = self.nodes[str(innovNum)]
            if int(innovNum) < 10:
                innovNum = " " + innovNum
            print(" ", innovNum, "|", "Node", node.getID(), "|", node.getType())
        print()

        print("CONNECTION GENES:")
        for innovNum in sorted(self.getConnectionGenes().keys(), key=lambda s: int(s)):
            connection = self.connections[innovNum]

            if int(innovNum) < 10:
                innovNum = " " + innovNum
            print(" ", innovNum, "|", connection.getInNode(), "->", connection.getOutNode(), "|", "Weight =", round(connection.getWeight(), 2), "|", "Enabled =", connection.isExpressed())

        print()

        print("--------------------------------------------------")
        print()

        return None

class ConnectionGene:
    def __init__(self, inNode, outNode, weight, expressed):
        self.inNode = int(inNode)
        self.outNode = int(outNode)
        self.weight = float(weight)
        self.expressed = bool(expressed)
        self.innovation = int(assignInnovationCon())

    def getInNode(self):
        return self.inNode

    def getOutNode(self):
        return self.outNode

    def getWeight(self):
        return self.weight

    def setWeight(self, weight):
        self.weight = weight

    def isExpressed(self):
        return self.expressed

    def getInnovation(self):
        return self.innovation

    def disable(self):
        self.expressed = False

    def enable(self):
        self.expressed = True

class NodeGene:
    def __init__(self, Type, ID):
        self.Type = str(Type)
        self.ID = int(ID)
        self.innovation = int(assignInnovationNod())

    def getType(self):
        return self.Type

    def getID(self):
        return self.ID

    def getInnovation(self):
        return self.innovation

class Species:
    def __init__(self, generation):
        self.ID = int(assignSpeciesID())
        self.created = int(generation)
        self.lastImproved = int(generation)
        self.members = {}
        self.representative = None
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = copy.deepcopy(representative)
        for member in members:
            self.members[str(copy.deepcopy(member.ID))] = copy.deepcopy(member)

    def getFitnesses(self):
        return [member.fitness for member in self.members.values()]

def assignInnovationNod():
    global innovationNod
    innovationNod += 1
    return innovationNod

def assignInnovationCon():
    global innovationCon
    innovationCon += 1
    return innovationCon

def assignSpeciesID():
    global speciesID
    speciesID += 1
    return speciesID