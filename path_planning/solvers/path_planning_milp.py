from typing import Any, Dict, List, Tuple, Set
import logging
import networkx as nx
from discrete_optimization.generic_tools.lp_tools import MilpSolver, MilpSolverName, \
    GurobiMilpSolver, CplexMilpSolver, ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from path_planning.path_planning import PathPlanning, PathSolution, NodeConstraints, Comparison

from gurobipy import Model, quicksum, Var, GRB
logger = logging.getLogger(__name__)

map_to_grb = {Comparison.UEQ: GRB.GREATER_EQUAL,
              Comparison.UP: GRB.GREATER_EQUAL,
              Comparison.LE: GRB.LESS_EQUAL,
              Comparison.LEQ: GRB.LESS_EQUAL,
              Comparison.EQ: GRB.EQUAL}


class LPPathPlanning_Gurobi(GurobiMilpSolver):
    def __init__(self, problem: PathPlanning, params_objective_function: ParamsObjectiveFunction = None):
        self.problem = problem
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.problem,
                                                       params_objective_function=params_objective_function)
        self.model = None
        self.variables: Dict[str, "Var"] = {}

    def compute_edges_in_and_out(self):
        graph: nx.DiGraph = self.problem.graph.to_networkx()
        edges = set(graph.edges())
        edges_index = set([(self.problem.index_nodes[e[0]],
                            self.problem.index_nodes[e[1]]) for e in edges])
        edges_in = {}
        edges_out = {}
        for e in edges_index:
            if e[0] not in edges_out:
                edges_out[e[0]] = set()
            if e[1] not in edges_in:
                edges_in[e[1]] = set()
            edges_in[e[1]].add(e)
            edges_out[e[0]].add(e)
        return edges_out, edges_in, edges_index

    def init_model(self, **kwargs: Any) -> None:
        edges_out, edges_in, edges_index = self.compute_edges_in_and_out()
        model = Model("path-planning")
        x = {e: model.addVar(obj=self.problem.matrix_length[e[0], e[1]],
                             lb=0, ub=1, vtype=GRB.BINARY, name=f"x_{e}") for e in edges_index}
        for node in self.problem.index_to_nodes:
            if node == self.problem.index_source:
                model.addLConstr(quicksum([x[e] for e in edges_out[node]]) == 1, name=f"outflow_{node}") # Outbound flow only.
                model.addLConstr(quicksum([x[e] for e in edges_in.get(node, [])]) == 0,  name=f"inflow_{node}") # Outbound flow only.
                continue
            if node == self.problem.index_sink:
                model.addLConstr(quicksum([x[e] for e in edges_out.get(node, [])]) == 0,
                                 name=f"outflow_{node}")  # No outbound
                model.addLConstr(quicksum([x[e] for e in edges_in.get(node, [])]) == 1,
                                 name=f"inflow_{node}")  # Inbound flow only.
                continue
            model.addLConstr(quicksum([x[e] for e in edges_in[node]]+[-x[e] for e in edges_out[node]]) == 0,
                             name=f"flow_conv_{node}")
        self.variables["x"] = x
        if not self.problem.node_constraints.is_empty():
            visit = {node: model.addVar(obj=0, vtype=GRB.BINARY, name=f"visit_{node}")
                     for node in self.problem.node_constraints.node_to_index_constraint}
            self.variables["visit"] = visit
            for node in visit:
                if node in {self.problem.index_sink, self.problem.index_source}:
                    model.addLConstr(visit[node] == 1)
                    continue
                else:
                    model.addLConstr(quicksum([x[e] for e in edges_in[node]]) == visit[node])
            for j in range(self.problem.node_constraints.nb_constraints):
                c: Tuple[List[int], Comparison, int] = self.problem.node_constraints.list_constraint[j]
                bound = c[2]

                if c[1] == Comparison.UP:
                    bound = c[2]+1
                if c[1] == Comparison.LE:
                    bound = c[2]-1
                model.addLConstr(quicksum([visit[n] for n in c[0]]),
                                 sense=map_to_grb[c[1]], rhs=bound,
                                 name=f"node_constraint_{j}")

        if not self.problem.path_constraints.is_empty():
            for j in range(self.problem.path_constraints.nb_constraints):
                c: Tuple[List[int], Comparison, int] = self.problem.path_constraints.list_constraint[j]
                bound = c[2]
                # TODO : dirty hardcoded things.
                if c[1] == Comparison.UP:
                    bound = c[2]+0.00001
                if c[1] == Comparison.LE:
                    bound = c[2]-0.00001
                model.addLConstr(quicksum([x[e]*self.problem.matrix_length_attr[c[0]][e[0], e[1]]
                                           for e in x]),
                                 sense=map_to_grb[c[1]], rhs=bound, name=f"path_constraint_{j}")
        self.model = model

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        list_solution_fits = []
        nb_solution = self.nb_solutions
        for j in range(nb_solution):
            logger.info(f"{j}-th solution has a cost of {self.get_obj_value_for_ith_solution(j)}")
            edges_selected = set()
            for e in self.variables["x"]:
                val = self.get_var_value_for_ith_solution(self.variables["x"][e], j)
                if val > 0.5:
                    edges_selected.add(e)
            _, nodes_path = self.rebuild_path(set_edges_selected=edges_selected)
            sol = PathSolution(problem=self.problem, path=nodes_path)
            fit = self.aggreg_sol(sol)
            list_solution_fits += [(sol, fit)]
        return ResultStorage(list_solution_fits=list_solution_fits,
                             mode_optim=self.params_objective_function.sense_function)

    def rebuild_path(self, set_edges_selected: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        edg_copy = set(set_edges_selected)
        edges_path = []
        nodes_path = [self.problem.index_source]
        while len(edg_copy) > 0:
            next_edge = [e for e in edg_copy if e[0] == nodes_path[-1]][0]
            edges_path += [next_edge]
            nodes_path += [next_edge[1]]
            edg_copy.remove(next_edge)
        return edges_path, nodes_path


