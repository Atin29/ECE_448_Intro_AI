from collections import OrderedDict

class node:
    def __init__(self, key, val=0):
        self.key = key
        self.val = val
        
        # dict mapping neighbor node key to edge weight, maintains order of insertion
        self.neighbors = OrderedDict()

    def __lt__(self, other):
        """For ordering nodes in a heap"""
        return self.key < other.key

    def __str__(self):
        return '(node: key {})'.format(self.key)

class graph:
    def __init__(self):
        # dict mapping key to node in the graph
        self.nodes = {}

    def adjacent(self, key1, key2):
        return key1 in self.nodes[key2].neighbors

    def neighbors(self, key):
        return self.nodes[key].neighbors

    def add_node(self, key, val=0):
        if key in self.nodes:
            raise ValueError('node with key `{}` already exists in graph'.format(key))
        self.nodes[key] = node(key, val)

    def remove_node(self, key):
        for connected_node_key in self.nodes[key].neighbors:
            self.nodes[connected_node_key].neighbors.pop(key)
        self.nodes.pop(key)

    def add_edge(self, key1, key2, weight=1):
        self.nodes[key1].neighbors[key2] = weight
        self.nodes[key2].neighbors[key1] = weight

    def remove_edge(self, key1, key2):
        self.nodes[key1].neighbors.pop(key2)
        self.nodes[key2].neighbors.pop(key1)

    def get_node_value(self, key):
        return self.nodes[key].val

    def set_node_value(self, key, val):
        self.nodes[key].val = val

    def get_edge_value(self, key1, key2):
        if self.adjacent(key1, key2):
            return nodes[key1].neighbors[key2]
        raise ValueError('no edge between nodes ({}) and ({})'.format(key1, key2))

    def set_edge_value(self, key1, key2, weight):
        if self.adjacent(key1, key2):
            self.nodes[key1].neighbors[key2] = weight
            self.nodes[key2].neighbors[key1] = weight
        else:
            raise ValueError('no edge between nodes ({}) and ({})'.format(key1, key2))

    def __str__(self):
        str_rep = []
        for key in self.nodes:
            node = self.nodes[key]
            neighbors_string = ', '.join([str(k) for k in node.neighbors])
            str_rep.append('{}: {}'.format(key, neighbors_string))
        return '\n'.join(str_rep)
