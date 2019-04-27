import sys, json

if __name__ == "__main__":
    graph = json.loads(sys.argv[1])
    print(graph)