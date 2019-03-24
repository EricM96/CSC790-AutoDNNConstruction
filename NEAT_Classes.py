# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:51:14 2019

@author: NathanLHall
"""

import copy

class Genome:
    def __init__(self):
        self.nodes = {}
        self.connections = {}
        self.fitness = None

    def addNodeGene(self, node):
        self.nodes[str(copy.deepcopy(node.getID()))] = copy.deepcopy(node)

    def getNodeGenes(self):
        return self.nodes

    def addConnectionGene(self, connection):
        self.connections[str(copy.deepcopy(connection.getInnovation()))] = copy.deepcopy(connection)

    def getConnectionGenes(self):
        return self.connections

    def setFitness(self, fitness):
        self.fitness = fitness

    def displayConnectionGenes(self):
#         for connection in self.connections.values():
        for innovNum in sorted(self.connections.keys()):
            print(innovNum, self.connections[innovNum].getInNode(), "->", self.connections[innovNum].getOutNode(), self.connections[innovNum].expressed)

class ConnectionGene:
    global innovation
    def __init__(self, inNode, outNode, weight, expressed, innovation):
        self.inNode = inNode # int
        self.outNode = outNode # int
        self.weight = weight # float
        self.expressed = expressed # boolean
        self.innovation = innovation # int
        innovation += 1

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
        self.Type = Type
        self.ID = ID

    def getType(self):
        return self.Type

    def getID(self):
        return self.ID