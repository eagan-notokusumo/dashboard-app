import sys
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvis.network as Networkx

class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph) # updates the graph with instances

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:  # if adjacent node of node is not found
                    graph[adjacent_node][node] = value # set the value of the node of the adjacent node to value

        return graph

    def get_nodes(self):
        return self.nodes

    def get_outgoing_edges(self, node):
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False: # checks if the node exists in the graph
                connections.append(out_node)
        
        return connections

    def value(self, node1, node2):
        return self.graph[node1][node2]

def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    shortest_path = {} # best cost of visiting each node in the graph from start_node
    previous_nodes = {} # stores the incoming trajectory of the best cost path for each node
    storagePath = []
    storagePrevious = []
    max_value = sys.maxsize # defines max value to be infinity

    for node in unvisited_nodes:
        shortest_path[node] = max_value # sets each node initially to be infinite cost.

    shortest_path[start_node] = 0 # sets the starting node to be zero.

    while unvisited_nodes: # Dijkstra algorithm runs until it visits all nodes in a graph.
        current_min_node = None

        for node in unvisited_nodes:
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        neighbors = graph.get_outgoing_edges(current_min_node)

        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor) # check initial value
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value # store all routes if possible?
                previous_nodes[neighbor] = current_min_node
                # storagePath.append(shortestPath[neighbor])
                # storagePrevious.append(previousNodes[neighbor])

        unvisited_nodes.remove(current_min_node)
    
    # print(storagePath)
    # print(storagePrevious)
    # print(shortestPath)
    # print(previousNodes)
    return previous_nodes, shortest_path

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
df = pd.read_csv('test_data.csv', encoding='UTF-8', delimiter=',')

df['OA'] = df['OA'].replace('\.', '', regex=True).astype(int)
df['OA/M3'] = df['OA'].div(df['Kapasitas'].values).astype(int)
df['Index'] = df['Tppt'].astype(str) + df['Tipe_Kendaraan'] + df['Tujuan']
groups = df.groupby(['Index','Tujuan','Product_Code']).min().apply(list)
df = groups.reset_index()
# print(groups)

src = ['Source']
vehicles = [node for node in df['Index']]
dstns = [node for node in df['Tujuan'].unique()]

nodes = src + vehicles + dstns

init_graph = {}
for node in nodes:
    init_graph[node] = {}

for vehicle in vehicles:
    init_graph[nodes[0]][vehicle] = 1

# print(init_graph)
for dstn in dstns:
    filtered= df[df['Tujuan'] == dstn]
    # print(query)
    query_vehicle = [node for node in filtered['Index']]
    print(query_vehicle)
    for vehicle in query_vehicle:
        index = query_vehicle.index(vehicle)
        # print(index)
        # print(vehicle)
        # print(q)
        init_graph[vehicle][dstn] = filtered.iloc[index, -1]
    # print(init_graph)

# print(init_graph)
# print(nodes)
# graph = Graph(nodes, init_graph)
# G = nx.DiGraph()
# G.add_nodes_from(nodes)
# G.add_weighted_edges_from(edges)


# pos = nx.nx_pydot.graphviz_layout(G)

# nx.draw(G, pos, with_labels=True, font_weight='bold')
# edge_weight = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_weight)

# pos = nx.random_layout(G)
# nx.draw(G, pos, with_labels=True, font_weight='bold')
# plt.show()

# previousNodes, shortestPath = dijkstra_algorithm(graph=graph, startNode='Source')

# printResults(previousNodes,shortestPath, startNode='Source', targetNode='BDG1')
#TODO: check runtime