import networkx as nx
import jsonpickle as jp
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

_is_ref = lambda s: isinstance(s, str) and s.startswith('@')
_is_int = lambda s: (s.startswith('-') and s[1:].isdigit()) or s.isdigit()
_to_id = partial(_per_cell, f = lambda v: '@%s'%(v.id if isinstance(v.node, dict) else v.node))


def _graph(Graph):
    """
    networkx graph with a quicker interface and support to adding cells to it as well deleting nodes and edges en-masse
    This is implemented on all networkx graphs so we implement it as a function of Class    
    """
    class _Graph(Graph):

        @classmethod
        def _key(self, key):
            return key
        
        def __getitem__(self, key):
            key = self._key(key)
            if _is_edge(key):
                return _as_value(self.edges[key[0], key[1]])
            else:
                return _as_value(self.nodes[key])

        def __setitem__(self, key, value):
            key = self._key(key)
            d = _as_dict(value)
            if _is_edge(key):
                self.add_edge(key[0], key[1], **d)
            else:
                self.add_node(key, **d)
                
        def __delitem__(self, key):
            key = self._key(key)
            if _is_edge(key):
                self.remove_edge(key[0], key[1])
            else:
                self.remove_node(key)

        def __contains__(self, key):
            key = self._key(key)
            return key in self.edges if _is_edge(key) else key in self._node
                
        def remove_node(self, node = None):
            if node is None:
                node = list(self.nodes)
            if is_array(node):
                for n in node:
                    self.remove_node(n)
            else:
                super(_Graph, self).remove_node(node)
        
        def remove_edge(self, u = None, v = None):
            if u is None:
                u = list(self.edges)
            if v is None and is_array(u):
                for e in u:
                    self.remove_edge(*e)
            else:
                super(_Graph, self).remove_edge(u,v)

    _Graph.__doc__ = """Additional api over networkx graph:

    # g['node'] = node   # to set the nodes
    # g['node']          # to access the node
    # del g['node']      # to delete the node

    # g['a', 'b'] = edge   # to set the edge from a to b
    # g['a', 'b']          # to access the edge 
    # del g['a', 'b']      # to delete the edge
    
    """ + Graph.__doc__                
    return _Graph


def _from_id(value, graph):
    if is_array(value):
        return [_from_id(v, graph) for v in value]
    elif isinstance(value, str) and value.startswith('@'):
        s = value[1:]
        return graph[int(s) if _is_int(s) else s]
    else:
        return value



def _graph_with_cells(Graph):

    class _Graph(Graph):        
        @classmethod
        def _key(self, key):
            """
            converts key into the Hash handling:
                if key is an edge, returns the edge pair
                if key is a ref, i.e. @id, will convert to an id first and if an int, to an int
                if key is a cell itself, will use the cell.node. This allows graph[cell] = cell code
            """
            if _is_edge(key):
                return (self._key(key[0]), self._key(key[1]))
            if _is_ref(key):
                key = key[1:]
                if _is_int(key):
                    key = int(key)
            if isinstance(key, Cell):
                return self._key(key.node)
            return Hash(key)

        def __add__(self, cell):
            graph = self
            if isinstance(cell, Cell):
                graph[cell.node] = cell
                graph = graph + cell.function
                graph = graph + list(cell.args)
                graph = graph + list(cell.kwargs.values())
            elif is_array(cell):
                for c in cell:
                    graph = graph + c                 
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
                print (node, c)
                c.function = _from_id(c.function, result)
                c.args = _from_id(c.args, result)
                c.kwargs = {key : _from_id(value, result) for key, value in c.kwargs.items()}
                result[node] = c
            return result        
                
        def add_parents(self, cell=None, parent=None):
            """
            Loops through chosen cell and parents and add them to the edges
            """
            graph = self
            if cell is None:
                cell = [graph[c] for c in graph.nodes]
            if _is_ref(cell):
                cell = graph[cell]
            elif is_array(cell):
                for c in cell:
                    graph = graph.add_parents(c, parent)
            if isinstance(cell, Cell):
                if isinstance(parent, Cell) or _is_ref(parent):
                    graph[parent, cell] = {} # this is where the action is
                elif is_array(parent):
                    for p in parent:
                        graph = graph.add_parents(cell, p)
                elif parent is None:
                    graph = graph.add_parents(cell, cell.function)
                    graph = graph.add_parents(cell, list(cell.args))
                    graph = graph.add_parents(cell, list(cell.kwargs.values()))
            return graph

        def to_table(self):
            rs = Dictable(node_id = list(nx.topological_sort(self)))
            rs = rs(node = lambda node_id: self[node_id])
            rs = rs(function = lambda node: node.function)
            rs = rs(args = lambda node: node.args)
            rs = rs(kwargs = lambda node: node.kwargs)
            rs = rs(node = lambda node: node.node)
            return rs
        
        def __repr__(self):
            return self.to_table().__repr__()

        def __str__(self):
            return self.to_table().__str__()

        def toJson(self):
            """
            We drop all edge information encoding only the nodes and before that, de-reference the nodes
            """
            return jp.encode(self.to_id()._node)
        
        @classmethod
        def fromJson(cls, j):
            """
            We construct the graph from the minimal json with only the dict of nodes
            We then add the parental graph
            We finally dereference the nodes
            """
            minimal = {int(k) if _is_int(k) else k: v for k, v in jp.decode(j).items()}
            res = cls()
            for k, v in minimal.items():
                res[k] = v
            res = res.add_parents()
            res = res.from_id()
            return res

    _Graph.__doc__ = """Added feauture: keys are always hashed but can be accessed either by the hash or the original value, 
    Support for adding Cell to the graph, adding for each cell its parents automatically\n\n
    """ + Graph.__doc__
                  
    return _Graph

DAG = _graph(nx.DiGraph)
XCL = _graph_with_cells(DAG)
