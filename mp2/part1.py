from heapq import heappush, heappop
import random

WIDGET_COMPONENTS = ['AEDCA', 'BEACD', 'BABCE', 'DADBD', 'BECBD']
FACTORY_DISTANCES = {
    'A': {
        'B': 1064,
        'C': 673,
        'D': 1401,
        'E': 277
    },
    'B': {
        'C': 958,
        'D': 1934,
        'E': 337
    },
    'C': {
        'D': 1001,
        'E': 399
    },
    'D': {
        'E': 387
    },
    'E': {
    }
}

# Complete the distance pairs going from dest_factory to start_factory
for start_factory in FACTORY_DISTANCES:
    distances_to_others = FACTORY_DISTANCES[start_factory]
    for dest_factory in distances_to_others:
        FACTORY_DISTANCES[dest_factory][start_factory] = \
                FACTORY_DISTANCES[start_factory][dest_factory]


def solve_for_num_stops():
    remaining_components = []
    for widget_composition in WIDGET_COMPONENTS:
        remaining_components.append([ch for ch in widget_composition])

    # Node key is a tuple of tuples containing the remaining components for
    # each widget
    start_node_key = tuple((tuple(w) for w in remaining_components))
    # Goal state is a state in which all widgets are built
    # i.e., remaining_components is empty

    # Frontier holds tuples of
    # (evaluation function val, path cost, path, current_node)
    # path cost is just len(path) in the num_stops case
    frontier = [(0, 0, [], start_node_key)]

    num_nodes_expanded = 0
    while frontier:
        evaluation_function_val, path_cost, path, node_key = heappop(frontier)
        remaining_components = node_key

        if len(remaining_components) == 0:
            assert path_cost == evaluation_function_val
            print('Solution found after {} nodes expanded'.format(num_nodes_expanded))
            print('Length {}:\n  {}'.format(path_cost, path))
            return path, num_nodes_expanded
        else:
            num_nodes_expanded += 1

        next_components_chosen = []
        for widget_components_remaining in remaining_components:
            next_component = widget_components_remaining[0]

            if next_component not in next_components_chosen:
                next_components_chosen.append(next_component)

                child_node_key = create_child_node_key(next_component,
                                    remaining_components)

                child_path = path[:]
                child_path.append(next_component)

                # g(n)
                child_path_cost = len(child_path)
                # h(n)
                try:
                    child_heuristic_val = max(len(child_parts_rem) for child_parts_rem in child_node_key)
                except ValueError:
                    child_heuristic_val = 0
                # f(n) = g(n) + h(n)
                child_eval_func_val = child_path_cost + child_heuristic_val

                child_node = (child_eval_func_val, child_path_cost, child_path, tuple(child_node_key))
                heappush(frontier, child_node)

    return [], num_nodes_expanded


def create_child_node_key(next_component, remaining_components):
    """Generate the next remaining components for each widget given the current
    remaining components and the next component to manufacture.
    """
    child_node_key = []
    for widget_components_remaining in remaining_components:
        if widget_components_remaining[0] == next_component:
            widget_next_components_remaining = list(widget_components_remaining)
            widget_next_components_remaining.pop(0)
            if len(widget_next_components_remaining) > 0:
                child_node_key.append(tuple(widget_next_components_remaining))
        else:
            child_node_key.append(widget_components_remaining)
    return tuple(child_node_key)


def solve_for_distance():
    remaining_components = []
    for widget_composition in WIDGET_COMPONENTS:
        remaining_components.append([ch for ch in widget_composition])

    # Node key is a tuple (1) the current location and
    # (2) tuples containing the remaining components for each widget
    start_node_key = (None, tuple((tuple(w) for w in remaining_components)))
    # Goal state is a state in which all widgets are built
    # i.e., remaining_components is empty

    # Frontier holds tuples of
    # (evaluation function val, path cost, path, current_node)
    frontier = [(0, 0, [], start_node_key)]

    num_nodes_expanded = 0
    while frontier:
        evaluation_function_val, path_cost, path, node_key = heappop(frontier)
        curr_loc, remaining_components = node_key

        if len(remaining_components) == 0:
            assert path_cost == evaluation_function_val
            print('Solution found after {} nodes expanded'.format(num_nodes_expanded))
            print('Length {}:\n  {}'.format(path_cost, path))
            return path, num_nodes_expanded
        else:
            num_nodes_expanded += 1

        next_components_chosen = []
        for widget_components_remaining in remaining_components:
            next_component = widget_components_remaining[0]

            if next_component not in next_components_chosen:
                next_components_chosen.append(next_component)

                child_node_key = create_child_node_key(next_component,
                                    remaining_components)

                child_path = path[:]
                child_path.append(next_component)

                # g(n)
                if curr_loc is None:
                    child_path_cost = 0
                else:
                    child_path_cost = path_cost + FACTORY_DISTANCES[curr_loc][next_component]

                # h(n)
                # For each widget w remaining, calculate the cost to sequentially visit each
                # factory remaining for w. Estimate the cost remaining to the goal
                # state as the max cost among these costs.
                child_widgets_max_cost = None
                for child_widget_remaining_components in child_node_key:
                    widget_cost = 0
                    for loc_idx in range(len(child_widget_remaining_components)-1):
                        src_factory = child_widget_remaining_components[loc_idx]
                        dst_factory = child_widget_remaining_components[loc_idx+1]
                        widget_cost += FACTORY_DISTANCES[src_factory][dst_factory]
                    if child_widgets_max_cost is None or widget_cost > child_widgets_max_cost:
                        child_widgets_max_cost = widget_cost
                child_heuristic_val = 0 if child_widgets_max_cost is None else child_widgets_max_cost

                # f(n) = g(n) + h(n)
                child_eval_func_val = child_path_cost + child_heuristic_val

                child_node = (child_eval_func_val, child_path_cost, child_path,
                                (next_component, tuple(child_node_key)))
                heappush(frontier, child_node)

    return [], num_nodes_expanded


def check_solution(path):
    print('Checking solution...')
    remaining_components = []
    for widget_composition in WIDGET_COMPONENTS:
        remaining_components.append([ch for ch in widget_composition])

    for factory in path:
        for widget_components_remaining in remaining_components:
            if len(widget_components_remaining) != 0 \
                    and widget_components_remaining[0] == factory:
                widget_components_remaining.pop(0)
    is_solution_valid = True
    for widget in remaining_components:
        if len(widget) > 0:
            is_solution_valid = False
    if is_solution_valid:
        print('Solution is valid')
    else:
        print('WARNING: Solution is invalid')

def gen_rand_widget():
    widget_len_str = input('Enter widget length: ')
    widget_len = int(widget_len_str)
    widget_list = []

    widget_comp_list = ['A', 'B', 'C', 'D', 'E']
    count = 0
    while(count < 5):
        rand_char = lambda n: ''.join([random.choice(widget_comp_list) for i in xrange(n)])
        #generate random widget of input length
        rand_widget = rand_widget(widget_len)
        widget_list.append(rand_widget)
        i += 1

    return widget_list


def main():
    #path, num_nodes_expanded = dfs_solve_for_num_stops()
    #check_solution(path)
    path, num_nodes_expanded = solve_for_num_stops()
    check_solution(path)
    #path, num_nodes_expanded = dfs_solve_for_distance()
    #check_solution(path)
    path, num_nodes_expanded = solve_for_distance()
    check_solution(path)

if __name__ == '__main__':
    main()
