import networkx as nx
from mombai._cell import Cell, Hash, _per_cell
from mombai._containers import as_list, is_array
from mombai._dictable import Dictable
from functools import partial
from copy import copy, deepcopy

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
    """
    class _Graph(Graph):

        @classmethod
        def key(self, key):
            return key
        
        def __getitem__(self, key):
            key = self.key(key)
            if _is_edge(key):
                return _as_value(self.edges[key[0], key[1]])
            else:
                return _as_value(self.nodes[key])

        def __setitem__(self, key, value):
            key = self.key(key)
            d = _as_dict(value)
            if _is_edge(key):
                self.add_edge(key[0], key[1], **d)
            else:
                self.add_node(key, **d)
                
        def __delitem__(self, key):
            key = self.key(key)
            if _is_edge(key):
                self.remove_edge(key[0], key[1])
            else:
                self.remove_node(key)
                
    _Graph.__doc__ = """Additional api over networkx graph:

    # g['node'] = node   # to set the nodes
    # g['node']          # to access the node
    # del g['node']      # to delete the node

    # g['a', 'b'] = edge   # to set the edge from a to b
    # g['a', 'b']          # to access the edge 
    # del g['a', 'b']      # to delete the edge
    
    """ + Graph.__doc__                
    return _Graph


_to_id = partial(_per_cell, f = lambda v: 'Cell[%i]'%v.id)

def _from_id(value, graph):
    if is_array(value):
        return [_from_id(v, graph) for v in value]
    elif isinstance(value, str) and value.startswith('Cell[') and value.endswith(']'):
        return graph[int(value[5:-1])]
    else:
        return value

def _to_id(value):
    if is_array(value):
        return [_to_id(v) for v in value]
    elif isinstance(value, Cell):
        return 'Cell[%i]'%value.id
    else:
        return value


def _graph_with_cells(Graph):

    class _Graph(Graph):        
        @classmethod
        def key(self, key):
            return (Hash(key[0]), Hash(key[1])) if _is_edge(key) else Hash(key)
        
        def add_parent(self, child, parent):
            graph = self
            if isinstance(parent, Cell):
                graph = graph + parent
                graph[parent.node, child.node] = {}
            elif is_array(parent):
                for p in parent:
                    graph.add_parent(child, p)
            return graph
        
        def to_id(self):
            result = deepcopy(self)
            for node in result.nodes:
                c = copy(result[node])
                c.function = _to_id(c.function)
                c.args = _to_id(c.args)
                c.kwargs = {key : _to_id(value) for key, value in c.kwargs.items()}
                result[node] = c
            return result
        
        def from_id(self):
            result = copy(self)
            for node in nx.topological_sort(self):
                c = result[node]
                c.function = _from_id(c.function, result)
                c.args = _from_id(c.args, result)
                c.kwargs = {key : _from_id(value, result) for key, value in c.kwargs.items()}
                result[node] = c
            return result        
                
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
        
        def to_table(self):
            rs = Dictable(node_id = list(nx.topological_sort(self)))
            rs = rs(node = lambda node_id: self[node_id])
            rs = rs(function = lambda node: node.function)
            rs = rs(args = lambda node: node.args)
            rs = rs(kwargs = lambda node: node.kwargs)
            rs = rs(node = lambda node: node.node)
            return rs
        
        def __rep__(self):
            return self.to_table().__repr__()

        def __str__(self):
            return self.to_table().__str__()
            
    _Graph.__doc__ = """Added feauture: keys are always hashed but can be accessed either by the hash or the original value, 
    Support for adding Cell to the graph, adding for each cell its parents automatically\n\n
    """ + Graph.__doc__
                  
    return _Graph

DAG = _graph(nx.DiGraph)
XCL = _graph_with_cells(DAG)

