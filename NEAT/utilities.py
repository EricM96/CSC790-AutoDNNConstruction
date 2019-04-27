"""
@Author: Eric McCullough
"""
from NEAT_Classes import Genome, ConnectionGene, NodeGene

def create_graph(solution):
    """
    @params: a single solution from the genetic algorithm
    @return: a dictionary representation of the solution's graph 
    """
    graph = {}

    # Iterate over nodes and instantiate their entries in the graph
    # Entry format is as follows: 'NodeID': (inputList, outputList)
    for key, value in solution.nodes.items():
        graph[key] = ([], []) 

    # Iterate over connections and add them to the graph entries 
    for key, value in solution.connections.items():
        graph[str(value.outNode)][0].append(str(value.inNode)) 
        graph[str(value.inNode)][1].append(str(value.outNode))

    return graph