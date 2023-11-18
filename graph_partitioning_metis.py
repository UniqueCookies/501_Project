import networkx as nx
import metis

adjacency_list = [
    [(1, 1), (4, 1)],  # Node 0 is connected to nodes 1 and 4 with weights 1
    [(0, 1), (2, 1)],  # Node 1 is connected to nodes 0 and 2 with weights 1
    [(1, 1), (3, 1)],  # Node 2 is connected to nodes 1 and 3 with weights 1
    [(2, 1), (4, 1)],  # Node 3 is connected to nodes 2 and 4 with weights 1
    [(0, 1), (3, 1)]   # Node 4 is connected to nodes 0 and 3 with weights 1
]

graph_to_test = metis.adjlist_to_metis(adjacency_list)

cuts, parts = metis.part_graph(graph_to_test, 2, recursive=False, dbglvl=metis.METIS_DBG_ALL)