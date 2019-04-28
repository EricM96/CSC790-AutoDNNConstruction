"""
@Author: Eric McCullough
@Description: this module takes a graph representation of a solution generated
              by the GA code, creates a PyTorch neural network from those features
              and determines fitness by running it on training and validation data
"""
import sys, json
import numpy as np
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F 

class Solution(nn.Module):
    def __init__(self, graph):
        super(Solution, self).__init__()

        self.features = self._create_model(graph)
        self.activation = nn.ReLU() 

        self.activation_graph = self.populate_activation_graph(graph)

    def _create_model(self, graph):
        """
        @params: a graph representation of a neural network
        @returns: a nn.Module for the graph 
        """
        
        in_nodes = [key for key, value in graph.items() if len(value[0]) == 0]
        out_nodes = [key for key, value in graph.items() if len(value[1]) == 0]
        hidden_nodes = [key for key, value in graph.items() if len(value[0]) != 0 and len(value[1]) != 0]

        moduleDict = nn.ModuleDict({}) 

        for node in in_nodes:
            moduleDict[node] = nn.Linear(1, len(graph[node][1]))

        for node in hidden_nodes:
            moduleDict[node] = nn.Linear(len(graph[node][0]), len(graph[node][1]))

        for node in out_nodes:
            moduleDict[node] = nn.Linear(len(graph[node][0]), 1)

        return moduleDict

    def populate_activation_graph(self, graph):
        """
        @params: a graph representation of a neural network
        @return: an activation graph, in the following format: 
        {node id: {
            input tensor: the input the node will work on
            node type: input or output or hidden
            dependencies: nodes that must execute before the current node can run
            }
        }
        """

        activation_graph = {}

        for key in graph.keys():
            input_len = len(graph[key][0])
            output_len = len(graph[key][1])

            if input_len == 0:
                node_type = "input"
            elif output_len == 0:
                node_type = "output"
            else:
                node_type = "hidden"

            activation_graph[key] = {'input tensor' : 
                np.zeros(input_len) if input_len > 0 else np.array([0.]),
                'node type' : node_type,
                'output nodes': graph[key][1],
                'dependencies': graph[key][0]}

        return activation_graph

    def feed_forward(self, X):
        """
        @params: network input
        @return: network prediction 
        """

        print(self.activation_graph)
        print()
        self._populate_inputs(X)
        print(self.activation_graph)

    def _populate_inputs(self, X):
        """
        @params: an input tensor for the network
        @return: the activation graph with a single value from the network input 
                 placed into each of input tensors for the input nodes 
        """

        i = 0
        for key, value in self.activation_graph.items():
            if value['node type'] == 'input':
                self.activation_graph[key]['input tensor'][0] = X[0][i]
                i += 1
        




if __name__ == "__main__":
    # graph = json.loads(sys.argv[1])
    graph = {'1': [[], ['4']], '3': [[], ['5']], '2': [[], ['4', '5']], '5': [['3', '2', '4'], []], '4': [['1', '2'], ['5']]}
    print(graph) 
    model = Solution(graph)
    model.feed_forward(np.random.rand(1, 3))
