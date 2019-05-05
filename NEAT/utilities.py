"""
@Author: Eric McCullough
"""
from NEAT_Classes import Genome, ConnectionGene, NodeGene
from subprocess import Popen, PIPE
import json, sys

def create_graph(solution):
    """
    @params: a single solution from the genetic algorithm
    @return: a dictionary representation of the solution's graph 
    """
    graph = {}

    # Iterate over nodes and instantiate their entries in the graph
    # Entry format is as follows: 'NodeID': (inputList, outputList)
    for node in solution.nodes:
        graph[str(node.ID)] = ([], [])

    # Iterate over connections and add them to the graph entries 
    for key, value in solution.connections.items():
        if value.expressed == True:
            graph[str(value.outNode)][0].append(str(value.inNode))
            graph[str(value.inNode)][1].append(str(value.outNode))

    print(graph)

    return graph

def compute_fitness(solution):
    solution_graph = create_graph(solution)

    # create shell command
    arg1 = sys.executable
    #arg2 = '/home/eam96/Documents/CSC790-AutoDNNConstruction/NEAT/fitness.py'
    arg2 = 'C:\\Users\\NathanLHall\\Desktop\\CSC 790 - Deep Learning\\CSC790-AutoDNNConstruction\\NEAT\\fitness.py'
    # arg2 = '/home/eam96/Desktop/CSC790-AutoDNNConstruction/NEAT/fitness.py'
    arg3 = json.dumps(solution_graph)
    # run shell command
    p = Popen([arg1, arg2, arg3], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    output, err = p.communicate()

    if len(err) != 0:
        print("Something went wrong")
        print(err)
        return -1.


    return float(output)