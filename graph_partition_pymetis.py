import numpy as np
import pymetis
import set_up

# https://pypi.org/project/graph-partition/

adjacency_list = [np.array([4, 2, 1]),
                  np.array([0, 2, 3]),
                  np.array([4, 3, 1, 0]),
                  np.array([1, 2, 5, 6]),
                  np.array([0, 2, 5]),
                  np.array([4, 3, 6]),
                  np.array([5, 3])]


n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency_list)  # 3 [1, 1, 1, 0, 1, 0, 0]
print(n_cuts, membership)
flipped_array = [1 if val == 0 else 0 for val in membership]
print(flipped_array)
# return number of cuts have to do, membership: same number = same group
nodes_part_0 = np.argwhere(np.array(membership) == 0).ravel()  # [3, 5, 6]
nodes_part_1 = np.argwhere(np.array(membership) == 1).ravel()  # [0, 1, 2, 4]
