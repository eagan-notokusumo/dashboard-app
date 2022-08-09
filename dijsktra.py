import sys
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvis.network as Networkx

class Graph(object):
    def __init__(self, nodes, initGraph):
        self.nodes = nodes
        self.graph = self.constructGraph(nodes, initGraph)

    def constructGraph(self, nodes, initGraph):
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(initGraph) # updates the graph with instances

        for node, edges in graph.items():
            for adjacentNode, value in edges.items():
                if graph[adjacentNode].get(node, False) == False:  # if adjacent node of node is not found
                    graph[adjacentNode][node] = value # set the value of the node of the adjacent node to value

        return graph

    def getNodes(self):
        return self.nodes

    def getOutgoingEdges(self, node):
        connections = []
        for outNode in self.nodes:
            if self.graph[node].get(outNode, False) != False: # checks if the node exists in the graph
                connections.append(outNode)
        
        return connections

    def value(self, node1, node2):
        return self.graph[node1][node2]

def dijkstraAlgorithm(graph, startNode):
    unvisitedNodes = list(graph.getNodes())

    shortestPath = {} # best cost of visiting each node in the graph from start_node
    previousNodes = {} # stores the incoming trajectory of the best cost path for each node
    storagePath = []
    storagePrevious = []
    maxValue = sys.maxsize # defines max value to be infinity

    for node in unvisitedNodes:
        shortestPath[node] = maxValue # sets each node initially to be infinite cost.

    shortestPath[startNode] = 0 # sets the starting node to be zero.

    while unvisitedNodes: # Dijkstra algorithm runs until it visits all nodes in a graph.
        currentMinNode = None

        for node in unvisitedNodes:
            if currentMinNode == None:
                currentMinNode = node
            elif shortestPath[node] < shortestPath[currentMinNode]:
                currentMinNode = node

        neighbors = graph.getOutgoingEdges(currentMinNode)

        for neighbor in neighbors:
            tentativeValue = shortestPath[currentMinNode] + graph.value(currentMinNode, neighbor) # check initial value
            if tentativeValue < shortestPath[neighbor]:
                shortestPath[neighbor] = tentativeValue # store all routes if possible?
                previousNodes[neighbor] = currentMinNode
                storagePath.append(shortestPath[neighbor])
                storagePrevious.append(previousNodes[neighbor])

        unvisitedNodes.remove(currentMinNode)
    
    print(storagePath)
    print(storagePrevious)
    print(shortestPath)
    print(previousNodes)
    return previousNodes, shortestPath

def printResults(previousNodes, shortestPath, startNode, targetNode):
    path = []
    node = targetNode

    while node != startNode:
        path.append(node)
        node = previousNodes[node]
    
    path.append(startNode)

    print('Best path value: {}'.format(shortestPath[targetNode]))
    print(" -> ".join(reversed(path)))

# TODO: extract sources from a CSV, join route and vehicle type in excel?
nodes = ['Source', '3082ME', '3082MF', '3082MG', '3082MH', '3082MI', '3082MJ', '3082MK', '3082ML', 'BDG1']
edges = [
    ('Source', '3082ME', 1),
    ('Source', '3082MF', 1),
    ('Source', '3082MG', 1),
    ('Source', '3082MH', 1),
    ('Source', '3082MI', 1),
    ('Source', '3082MJ', 1),
    ('Source', '3082MK', 1),
    ('Source', '3082ML', 1),
    ('3082ME', 'BDG1', 63920),
    ('3082MF', 'BDG1', 51807),
    ('3082MG', 'BDG1', 57571),
    ('3082MH', 'BDG1', 56520),
    ('3082MI', 'BDG1', 48889),
    ('3082MJ', 'BDG1', 55000),
    ('3082MK', 'BDG1', 58000),
    ('3082ML', 'BDG1', 60000),
    ]

initGraph = {}
for node in nodes:
    initGraph[node] = {}

# TODO: automatically input, maybe define initially as list of tuples of strings
initGraph['Source']['3082ME'] = 1
initGraph['Source']['3082MF'] = 1
initGraph['Source']['3082MG'] = 1
initGraph['Source']['3082MH'] = 1
initGraph['Source']['3082MI'] = 1
initGraph['Source']['3082MJ'] = 1
initGraph['Source']['3082MK'] = 1
initGraph['Source']['3082ML'] = 1

#TODO: automate OA/m3 entry
initGraph['3082ME']['BDG1'] = 63920
initGraph['3082MF']['BDG1'] = 51807
initGraph['3082MG']['BDG1'] = 57571
initGraph['3082MH']['BDG1'] = 56520
initGraph['3082MI']['BDG1'] = 48889
initGraph['3082MJ']['BDG1'] = 55000
initGraph['3082MK']['BDG1'] = 58000
initGraph['3082ML']['BDG1'] = 60000




graph = Graph(nodes, initGraph)
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges)


# pos = nx.nx_pydot.graphviz_layout(G)

# nx.draw(G, pos, with_labels=True, font_weight='bold')
# edge_weight = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_weight)

# pos = nx.random_layout(G)
# nx.draw(G, pos, with_labels=True, font_weight='bold')
# plt.show()

previousNodes, shortestPath = dijkstraAlgorithm(graph=graph, startNode='Source')

printResults(previousNodes,shortestPath, startNode='Source', targetNode='BDG1')
#TODO: check runtime