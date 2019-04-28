"""
@Author: Eric McCullough
@Description: this module takes a graph representation of a solution generated
              by the GA code, creates a PyTorch neural network from those features
              and determines fitness by running it on training and validation data
"""
import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

torch.set_default_tensor_type('torch.DoubleTensor')

class Solution(nn.Module):
    def __init__(self, graph):
        super(Solution, self).__init__()

        self.features = self._create_model(graph)
        self.activation = F.relu

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

        self._populate_inputs(X)
        self._step_inputs()
        self._step_hiddens()
        self._display_activation_graph()

    def _populate_inputs(self, X):
        """
        @description: a helper function for feed_forward
        @params: an input tensor for the network
        @return: the activation graph with a single value from the network input 
                 placed into each of input tensors for the input nodes 
        """

        i = 0
        for key, value in self.activation_graph.items():
            if value['node type'] == 'input':
                self.activation_graph[key]['input tensor'][0] = X[0][i]
                i += 1

    def _step_inputs(self):
        """
        @description: a helper function for feed_forward
        @params: none
        @return: runs the network feature associated with each input node and 
                 adds the output to the input tensor for node that depends on it
        """

        for key, value in self.activation_graph.items():
            if value['node type'] == 'input':
                t = torch.tensor(value['input tensor'])
                y_hat = self.features[key](t)

                for val, out_node in zip(y_hat, value['output nodes']):
                    i = 0
                    while True:
                        if self.activation_graph[out_node]['input tensor'][i] == 0.:
                            self.activation_graph[out_node]['input tensor'][i] = np.float64(val)
                            break
                        else:
                            i += 1

    def _step_hiddens(self):
        """
        @description: a helper function for feed_forward
        @params: none
        @return: runs the network feature associated with each hidden node and 
                 adds the output to the input tensor for its dependants 
        """

        hidden_keys = [key for key, val in self.activation_graph.items() if val['node type'] == 'hidden']
        visited_nodes = [key for key, val in self.activation_graph.items() if val['node type'] == 'input']
        print(hidden_keys)

        for node in hidden_keys:
            if set(self.activation_graph[node]['dependencies']) <= set(visited_nodes):
                t = torch.tensor(self.activation_graph[node]['input tensor'])
                y_hat = self.features[node](t)

                for val, out_node in zip(y_hat, self.activation_graph[node]['output nodes']):
                    i = 0
                    while True:
                        if self.activation_graph[out_node]['input tensor'][i] == 0.:
                            self.activation_graph[out_node]['input tensor'][i] = np.float64(val)
                            break
                        else:
                            i += 1

            else:
                hidden_keys.append(node)
                
    def _display_activation_graph(self):
        for key, value in self.activation_graph.items():
            print(key, value) 

        print()
        




if __name__ == "__main__":
    # graph = json.loads(sys.argv[1])
    graph = {'1': [[], ['4']], '3': [[], ['5']], '2': [[], ['4', '5']], '5': [['3', '2', '4'], []], '4': [['1', '2'], ['5']]}
    model = Solution(graph)
    model.feed_forward(np.random.rand(1, 3))
