import sys, json
import torch.nn as nn
import torch.nn.functional as F 

def id_features(graph):
    in_nodes = [key for key, value in graph.items() if len(value[0]) == 0]
    out_nodes = [key for key, value in graph.items() if len(value[1]) == 0]

    moduleDict = {} 

    for node in in_nodes:
        moduleDict[node] = nn.Linear(1, len(graph[node][1]))

    print(moduleDict) 

if __name__ == "__main__":
    graph = json.loads(sys.argv[1])
    print(graph) 
    id_features(graph) 