# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:41:39 2019

@author: NathanLHall
"""

from NEAT_Classes import Genome, ConnectionGene, NodeGene
from NEAT_Reproduction import addConnectionMutation, addNodeMutation, weightMutation, expressedMutation, crossover
import copy

def main():
    # Testing an example, as described in Figure 4 of NEAT paper
    node1 = NodeGene("INPUT", 1)
    node2 = NodeGene("INPUT", 2)
    node3 = NodeGene("INPUT", 3)
    node4 = NodeGene("HIDDEN", 4)
    node5 = NodeGene("OUTPUT", 5)
    node6 = NodeGene("HIDDEN", 6)
    nodes1 = [node1, node2, node3, node4, node5]
    nodes2 = [node1, node2, node3, node4, node5, node6]

    connection1 = ConnectionGene(1, 4, 1, True, 1)
    connection2 = ConnectionGene(2, 4, 1, True, 2)
    connection3 = ConnectionGene(3, 4, 1, True, 3)
    connection4 = ConnectionGene(2, 5, 1, False, 4)
    connection5 = ConnectionGene(3, 5, 1, True, 5)
    connection6 = ConnectionGene(4, 5, 1, True, 6)
    connection7 = ConnectionGene(1, 6, 1, True, 7)
    connection8 = ConnectionGene(6, 4, 1, True, 8)
    connections1 = [connection1, connection2, connection4, connection5, connection6]

    parent1 = Genome()
    parent2 = Genome()

    for node in nodes1:
        parent1.addNodeGene(node)

    for node in nodes2:
        parent2.addNodeGene(node)

    for connection in connections1:
        parent1.addConnectionGene(connection)

    connection1a = copy.deepcopy(connection1)
    connection1a.disable()
    connections2 = [connection1a, connection2, connection3, connection4, connection6, connection7, connection8]
    for connection in connections2:
        parent2.addConnectionGene(connection)

    parent1.setFitness(1)
    parent2.setFitness(1)

    offspring = crossover(parent1, parent2)

    offspring.displayConnectionGenes()

main()