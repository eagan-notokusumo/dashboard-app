from operator import itemgetter
import sys
import numpy as np
import pandas as pd
import itertools



class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph) # updates the graph with instances

        # for node, edges in graph.items():
        #     for adjacent_node, value in edges.items():
        #         if graph[adjacent_node].get(node, False) == False:  # if adjacent node of node is not found
        #             graph[adjacent_node][node] = value # set the value of the node of the adjacent node to value

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

    def remove_edge(self, node1, node2):
        return self.graph.pop([node1][node2])

def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    shortest_path = {} # best cost of visiting each node in the graph from start_node
    previous_nodes = {} # stores the incoming trajectory of the best cost path for each node
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

        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path

def path(previousNodes, startNode, targetNode):
    path = []
    node = targetNode

    while node != startNode:
        path.append(node)
        node = previousNodes[node]
    
    path.append(startNode)
    path = list(reversed(path))
    # print(path)
    return path

def yen_algorithm(init_graph, nodes, start_node, end_node, max_k):
    graph = Graph(nodes, init_graph)
    previous_nodes, shortest_path = dijkstra_algorithm(graph, start_node)

    A = [{
        'cost': shortest_path[end_node],
        'path_': path(previous_nodes, start_node, end_node)
    }]

    B = []

    if not A[0]['path_']:
        return A

    for k in range(1, max_k):
        for i in range(0,len(A[k-1]['path_']) - 1, 2):
            spur_node = A[k-1]['path_'][i]
            root_path = A[k-1]['path_'][:i]

            removed_edges = []
            pp = sorted([A[j]['path_'][:i] for j in range(len(A)-1)])
            pp_unique = list(pp for pp,_ in itertools.groupby(pp))

            for path_k in A:
                current_path = path_k['path_']
                # print(current_path[i+1])
                if len(current_path[i+1]) > 8 and current_path[i+1] in nodes: # removing transient nodes
                    # print(current_path[i+1])
                    nodes.remove(current_path[i+1])
                    # print(current_path[:i])
                    if len(current_path) - 1 > i and root_path == current_path[:i]:
                    # print(len(nodes))
                        cost = init_graph[current_path[i]].pop(current_path[i+1])
                        init_graph[current_path[i+1]].pop(current_path[i])
                        if cost == -1:
                            continue
                        removed_edges.append([current_path[i], current_path[i+1], cost])
                    
                elif any(pp_unique):
                    for previous_path in pp_unique:
                        p0 = len(previous_path[:i-2])
                        p1 = len(previous_path[:i-1])
                        if current_path[:i] not in pp_unique:
                            if previous_path[p1] not in init_graph[previous_path[p0]].keys():
                                continue
                            else:
                                cost2 = init_graph[previous_path[p0]].pop(previous_path[p1])
                                init_graph[previous_path[p1]].pop(previous_path[p0])
                                removed_edges.append([previous_path[p0], previous_path[p1], cost2])
                        elif current_path[:i] == root_path and previous_path != current_path[:i]:
                            if previous_path[p1] not in init_graph[previous_path[p0]].keys():
                                continue
                            else:
                                cost3 = init_graph[previous_path[p0]].pop(previous_path[p1])
                                init_graph[previous_path[p1]].pop(previous_path[p0])
                                removed_edges.append([previous_path[p0], previous_path[p1], cost3])

                    # print(removed_edges)
                    # print(init_graph)

                    # TODO: issue that nodes disappear

            graph_spur = Graph(nodes, init_graph)
            pn_init, sp_init = dijkstra_algorithm(graph_spur, start_node='Source')
            pn_spur, sp_spur = dijkstra_algorithm(graph_spur, spur_node)
            spur_path = {
                'cost': sp_spur[end_node],
                'path_': path(pn_spur, spur_node, end_node)
            }
            
            if spur_path['path_']:
                total_path = root_path + spur_path['path_']
                total_cost = sp_init[spur_node] + spur_path['cost'] #TODO: the issue is shortest path is the oldest version
                potential_k = {'cost': total_cost, 'path_': total_path}

                if not (potential_k in B):
                    B.append(potential_k)

            for edge in removed_edges:
                init_graph[edge[0]][edge[1]] = edge[2]
                init_graph[edge[1]][edge[0]] = edge[2]
                if edge[1] not in nodes:
                    nodes.append(edge[1])

        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break
    return A
                



