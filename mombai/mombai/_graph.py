import networkx as nx
from mombai._dict import Dictattr
from mombal._cell import Cell
from mombai._containers import as_list, is_array

def _as_dict(value):
    if isinstance(value, dict):
        return value
    else:
        return dict(value = value)

def _as_value(d):
    if len(d)==1 and 'value' in d.keys():
        return d['value']
    else:
        return d

def _is_edge(key):
    return isinstance(key, tuple) and len(key) == 2


def _graph(Graph):
    """
    networkx graph with a quicker interface and support to adding cells to it
    This is implemented on all networkx graphs so we implement it as a function of Class
    
    additional api:
    # g['node'] = node   # to set the nodes
    # g['node']          # to access the node
    # del g['node']      # to delete the node

    # g['a', 'b'] = edge   # to set the edge from a to b
    # g['a', 'b']          # to access the edge 
    # del g['a', 'b']      # to delete the edge

    """
    class _Graph(Graph):

        def __getitem__(self, key):
            if _is_edge(key):
                return _as_value(self.edges[key[0], key[1]])
            else:
                return _as_value(self.nodes[key])

        def __setitem__(self, key, value):
            d = _as_dict(value)
            if _is_edge(key):
                self.add_edge(key[0], key[1], **d)
            else:
                self.add_node(key, **d)
        def __delitem__(self, key):
            if _is_edge(key):
                self.remove_edge(key[0], key[1])
            else:
                self.remove_node(key)
                
        def add_parent(self, child, parent):
            graph = self
            if isinstance(parent, Cell):
                graph = graph + parent
                graph[parent.node, child.node] = {}
            elif is_array(parent):
                for p in parent:
                    graph.add_parent(child, p)
            return graph

        def __add__(self, cell):
            graph = self
            if isinstance(cell, Cell):
                graph[cell.node] = cell
                graph = graph.add_parent(cell, cell.function)
                graph = graph.add_parent(cell, list(cell.args))
                graph = graph.add_parent(cell, list(cell.kwargs.values()))
            elif is_array(cell):
                for c in cell:
                    graph = graph + c
            return graph
                   
    return _Graph

DAG = _graph(nx.DiGraph)
