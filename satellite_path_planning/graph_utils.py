from satellite_path_planning.problem import SppProblem, SppSolution
import networkx as nx
from igraph import Graph


def from_nx_to_igraph(graph: nx.DiGraph):
    g = Graph.from_networkx(graph)
    return g

def script():
    

def analyze_graphs(problem: SppProblem):
    problem.compute_transitive_closure_graph()
    graph = problem.graph
    closure_graph = nx.transitive_closure_dag(graph,
                                              topo_order=sorted(list(graph.nodes)))


    self.closure_graph: nx.DiGraph = graph
        #node_to_acquisition_descendants = self.problem.node_to_acquisition_descendants