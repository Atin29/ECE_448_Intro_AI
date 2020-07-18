import string
from collections import deque
from heapq import heappush, heappop

from graph import node, graph

OPEN_CHAR = ' '
START_POS_CHAR = 'P'
DOT_CHAR = '.'
WALL_CHAR = '%'
VISITED_CHAR = '-'
SOLUTION_PATH_CHAR = '.'
DOT_LABELS = string.digits + string.ascii_letters

class maze:
    def __init__(self, maze_file_name):
        # coordinates are stored (row index, column index)
        self.dot_locations = []
        self.start_coords = None

        # a 2D list of chars
        self.matrix = []
        self.graph = graph()
        node_keys = set()

        with open(maze_file_name) as f:
            for row_idx, row in enumerate(f):
                self.matrix.append([ch for ch in row.rstrip().replace('%',WALL_CHAR)])
                for col_idx, ch in enumerate(row):
                    if ch == WALL_CHAR: continue
                    coords = (row_idx, col_idx)
                    node_keys.add(coords)
                    self.graph.add_node(coords)
                    if ch == '.':
                        self.dot_locations.append(coords)
                    elif ch == 'P':
                        self.start_coords = coords

                    # Connect the new node to any existing nodes directly above or left
                    if (row_idx-1, col_idx) in node_keys:
                        existing_node_key = (row_idx-1, col_idx)
                        self.graph.add_edge(coords, existing_node_key)
                    if (row_idx, col_idx-1) in node_keys:
                        existing_node_key = (row_idx, col_idx-1)
                        self.graph.add_edge(coords, existing_node_key)

    def copy_matrix(self):
        """Return a deep copy of the matrix"""
        return [row[:] for row in self.matrix]

    def __str__(self):
        return '\n'.join(''.join(row) for row in self.matrix)


def solve_maze_astar(m, print_path_solutions=False, print_steps=False):
    # 1. Get distances between all pairs of dots
    # 2. Search for a sequence of visiting all dots that minimizes path length
    g = m.graph
    dot_locations = [m.start_coords] + m.dot_locations

    lowest_costs_btwn_dots = []

    # 1. Get distances between all pairs of dots
    all_pairs_dot_distances = {}
    for i in range(len(dot_locations)):
        start_coords = dot_locations[i]
        all_pairs_dot_distances[start_coords] = {}
        for j in range(i+1, len(dot_locations)):
            dest_coords = dot_locations[j]
            path, nodes_expanded = a_star_search(start_coords, dest_coords, g, print_path_solutions)
            path_len = len(path) - 1 # Subtract 1 because the path includes the start coords as a step

            if print_path_solutions:
                if print_steps:
                    for i in range(len(nodes_expanded)):
                        mat = m.copy_matrix()
                        for r, c in nodes_expanded[:i]:
                            mat[r][c] = VISITED_CHAR
                        for row in mat:
                            print(''.join(row))

                mat = m.copy_matrix()
                for r, c in nodes_expanded:
                    mat[r][c] = VISITED_CHAR
                for r, c in path:
                    mat[r][c] = SOLUTION_PATH_CHAR
                for row in mat:
                    print(''.join(row))
                print()

            all_pairs_dot_distances[start_coords][dest_coords] = path_len
            lowest_costs_btwn_dots.append((path_len, start_coords, dest_coords))
    lowest_costs_btwn_dots.sort()

    # Complete the pairs going from dest_coords to start_coords
    for start_coords in all_pairs_dot_distances:
        distances_to_others = all_pairs_dot_distances[start_coords]
        for dest_coords in distances_to_others:
            all_pairs_dot_distances[dest_coords][start_coords] = \
                    all_pairs_dot_distances[start_coords][dest_coords]

    print('Finished computing optimal paths between all pairs of dots')

    # 2. Search for a sequence of visiting all dots that minimizes path length
    # Define a node as (current coords, coords of dots remaining)
    start_node_key = (m.start_coords, tuple(sorted(m.dot_locations)))

    # Frontier holds tuples of (evaluation function val, path cost, path, current_node)
    frontier = [(0, 0, [], start_node_key)]

    print('Running A* search for paths')

    visited = {}
    found = False
    num_nodes_expanded = 0
    optimal_solution_path_cost = None
    while frontier:
        evaluation_function_val, path_cost, path, node_key = heappop(frontier)
        coords, dots_remaining = node_key
        path_cost = -path_cost

        # If there are no more nodes on the frontier with evaluation function val
        # less than the optimal solution path cost, stop the search
        if optimal_solution_path_cost and\
                evaluation_function_val > optimal_solution_path_cost:
            break

        # Since common states have the same value of h(n), if the search
        # encounters a state that it has seen before, it is guaranteed that
        # the path cost g(n) to get to that state is >= the cost of the
        # previously taken path to get to that state because the heap
        # ordered the nodes by f(n) = g(n) + h(n)
        if node_key in visited:
            assert path_cost >= visited[node_key]
            continue

        visited[node_key] = path_cost
        path.append(coords)

        if len(dots_remaining) == 0:
            assert path_cost == evaluation_function_val
            found = True
            if optimal_solution_path_cost is None\
                or path_cost < optimal_solution_path_cost:
                    optimal_solution_path_cost = path_cost
                    optimal_solution_path = path
            print('Found solution\ncost: {}\n{}\n'.format(path_cost, path))
        else:
            num_nodes_expanded += 1

        for next_coords_idx in range(len(dots_remaining)):
            next_dots_remaining = list(dots_remaining)
            next_dots_remaining.pop(next_coords_idx)
            next_coords = dots_remaining[next_coords_idx]
            next_node_key = (next_coords, tuple(next_dots_remaining))

            # g(n) is next_path_cost
            step_cost = all_pairs_dot_distances[coords][next_coords]
            next_path_cost = path_cost + step_cost

            # h(n) is est_cost_remaining_from_next
            # With k dots remaining for the child, estimate the cost remaining
            # as the sum of the k lowest cost paths between pairs of remaining dots
            # in the parent. This heuristic is admissible.
            est_cost_remaining_from_next = 0
            num_taken = 0
            i = 0
            while num_taken < len(next_dots_remaining):
                path_len, start_coords, dest_coords = lowest_costs_btwn_dots[i]
                if start_coords in dots_remaining and dest_coords in dots_remaining:
                    est_cost_remaining_from_next += path_len
                    num_taken += 1
                i += 1

            # f(n) = g(n) + h(n)
            next_eval_function_val = next_path_cost + est_cost_remaining_from_next

            # Negate next_path_cost to prioritize expanding paths that are already
            # more complete if there is a tie on next_eval_function_val.
            child_node = (next_eval_function_val, -next_path_cost, path[:], next_node_key)
            heappush(frontier, child_node)

    print('{} nodes expanded'.format(num_nodes_expanded))
    if found:
        print('Optimal solution through all dots: length {}'.format(optimal_solution_path_cost))
        print(optimal_solution_path)
        mat = m.copy_matrix()
        for i, (r, c) in enumerate(optimal_solution_path):
            mat[r][c] = DOT_LABELS[i]
        for row in mat:
            print(''.join(row))
    else:
        print('No solution found')


def dfs(start_node_key, dest_node_key, graph, print_solution=True):
    """Search for a path from start_node_key to dest_node_key in graph
    using a stack for the frontier.
    """
    return search(start_node_key, dest_node_key, graph,
                list, lambda l: l.pop(), lambda l, e: l.append(e),
                print_solution=print_solution)

def bfs(start_node_key, dest_node_key, graph, print_solution=True):
    """Search for a path from start_node_key to dest_node_key in graph
    using a queue for the frontier.
    """
    return search(start_node_key, dest_node_key, graph,
                deque, lambda q: q.popleft(), lambda q, e: q.append(e),
                print_solution=print_solution)

def greedy_best_first_search(start_node_key, dest_node_key, graph, print_solution=True):
    """Search for a path from start_node_key to dest_node_key in graph
    using a priority queue (heap) for the frontier.
    The greedy search heuristic orders coords in the priority queue
    by manhattan distance to the destination.
    """
    manhattan_dist = lambda row, col: abs(row - dest_node_key[0]) + abs(col - dest_node_key[1])
    greedy_search_heuristic = lambda node_key, path: manhattan_dist(node_key[0], node_key[1])
    return search(start_node_key, dest_node_key, graph,
                list, lambda pq: heappop(pq), lambda pq, e: heappush(pq, e),
                greedy_search_heuristic,
                print_solution=print_solution)

def a_star_search(start_node_key, dest_node_key, graph, print_solution=True):
    """Search for a path from start_node_key to dest_node_key in graph
    using a priority queue (heap) for the frontier.
    The A* search heuristic orders coords in the priority queue
    by the sum of manhattan distance to the destination and
    length of path taken to get to those coords.
    """
    manhattan_dist = lambda row, col: abs(row - dest_node_key[0]) + abs(col - dest_node_key[1])
    a_star_heuristic = lambda node_key,path: manhattan_dist(node_key[0], node_key[1]) + len(path)
    return search(start_node_key, dest_node_key, graph,
                list, lambda pq: heappop(pq), lambda pq, e: heappush(pq, e),
                a_star_heuristic,
                print_solution=print_solution)


def search(start_node_key, dest_node_key, graph,
        search_structure, structure_get, structure_put,
        heuristic=(lambda node_key, path: 0),
        print_solution=True):
    """Searches for a path from start_node_key to dest_node_key in graph.
    Specify the frontier data structure using parameters search_structure,
    structure_get, and structure_put. If using a heapq for the frontier, specify
    a heuristic function of the form \lambda node_key, path: int\ to determine
    node expansion order.
    """
    # Structure from which we pull search nodes based on the given heuristic.
    frontier = search_structure(
                    [(heuristic(start_node_key, []), start_node_key, [])]
                )

    visited = set()
    nodes_expanded = []

    while frontier:
        _, node_key, path = structure_get(frontier)
        if node_key in visited:
            continue

        visited.add(node_key)
        path.append(node_key)
        nodes_expanded.append(node_key)

        if node_key == dest_node_key:
            if print_solution:
                print('Solution from {} to {} found'.format(start_node_key, dest_node_key))
                print('{} nodes expanded'.format(len(nodes_expanded)))
                print('path length: {}'.format(len(path)-1))
            return path, nodes_expanded

        # Note the [:] to deep copy the path list
        children = [(heuristic(neighbor_key, path), neighbor_key, path[:])
                    for neighbor_key in graph.neighbors(node_key) if neighbor_key not in visited]

        for n in children:
            structure_put(frontier, n)

    return None, nodes_expanded


def main():
    maze_file_dir = 'mazes'
    part_1_1_file_names = ['mediumMaze.txt', 'bigMaze.txt', 'openMaze.txt']
    part_1_2_file_names = ['tinySearch.txt', 'smallSearch.txt', 'mediumSearch.txt']

    search_functions = [('DFS', dfs), ('BFS', bfs),
                        ('Greedy Best First Search', greedy_best_first_search),
                        ('A Star Search', a_star_search)]

    for fname in part_1_1_file_names:
        m = maze('{}/{}'.format(maze_file_dir, fname))
        print('_________________{}_________________'.format(fname))
        for search_name, search_function in search_functions:
            print(search_name)
            path, nodes_expanded = search_function(m.start_coords, m.dot_locations[0], m.graph)
            mat = m.copy_matrix()
            for r, c in path:
                mat[r][c] = SOLUTION_PATH_CHAR
            for row in mat:
                print(''.join(row))
            print()
        print()

    for fname in part_1_2_file_names:
        m = maze('{}/{}'.format(maze_file_dir, fname))
        print('_________________{}_________________'.format(fname))
        solve_maze_astar(m, print_path_solutions=False, print_steps=False)
        print()

if __name__ == '__main__':
    main()
