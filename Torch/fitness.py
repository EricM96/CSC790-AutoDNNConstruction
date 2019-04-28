import sys, json
import torch.nn as nn
import torch.nn.functional as F 

def id_features(graph):
    in_nodes = [key for key, value in graph.items() if len(value[0]) == 0]
    out_nodes = [key for key, value in graph.items() if len(value[1]) == 0]
    hidden_nodes = [key for key, value in graph.items() if len(value[0]) != 0 and len(value[1]) != 0]

    print("Input nodes: ", in_nodes)
    print("Hidden nodes", hidden_nodes)
    print("out_nodes", out_nodes)

    moduleDict = {} 

    for node in in_nodes:
        moduleDict[node] = nn.Linear(1, len(graph[node][1]))

    for node in hidden_nodes:
        moduleDict[node] = nn.Linear(len(graph[node][0]), len(graph[node][1]))

    for node in out_nodes:
        moduleDict[node] = nn.Linear(len(graph[node][0]), 1)

    print(moduleDict) 

if __name__ == "__main__":
    graph = json.loads(sys.argv[1])
    print(graph) 
    id_features(graph) 