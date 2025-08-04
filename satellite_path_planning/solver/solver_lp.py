from __future__ import annotations

from typing import Any, Callable, Sequence

#from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from satellite_path_planning.solver.spp_solver import SppSolver
from satellite_path_planning.problem import SppSolution

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, LinExpr, Model, Var

from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver


class BaseGurobiSppSolver(SppSolver, GurobiMilpSolver):
    vars_node: Sequence[Var]

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> SppSolution:

        list_nodes = []

        for i in range(0, self.problem.nb_nodes):
            if get_var_value_for_current_solution(self.vars_node[i]) > 0.5:
                list_nodes.append(self.problem.ind_to_id[i])

        return SppSolution(self.problem, list_nodes)

    def set_warm_start(self, solution: SppSolution) -> None:
        """Make the solver warm start from the given solution."""
        for i in range(0, self.problem.nb_nodes):
            if self.problem.ind_to_id[i] in solution.solution_path:
                self.vars_node[i].Start = 1
            else:
                self.vars_node[i].Start = 0


class SppMilpSolver(BaseGurobiSppSolver):
    def init_model(self, **kwargs: Any) -> None:

        # Create a new model
        self.model = Model()

        # Create variables

        nb_nodes = self.problem.graph.number_of_nodes()
        nb_edges = self.problem.graph.number_of_edges()

        vars_node = self.model.addVars(nb_nodes, vtype=GRB.BINARY, name="N")
        vars_edge = self.model.addVars(nb_edges, vtype=GRB.BINARY, name="E")

        # Set objective
        obj_exp = LinExpr()
        obj_exp.addTerms([self.problem.acquisition_dict[self.problem.graph.nodes[node]["acquisition_request"]] for node in self.problem.graph.nodes], vars_node.select())  # TODO : value des noeuds : remplacer 1
        self.model.setObjective(obj_exp, GRB.MAXIMIZE)

        # each request must be chosen at most one time
        list_exp = []

        for i in range(len(self.problem.acquisition_dict)):
            list_exp.append(LinExpr())

        for node in self.problem.graph.nodes:
            list_exp[node["acquisition_request_id"]].addTerms(1, vars_node[self.problem.id_to_ind[node]])

        for i in range(len(self.problem.acquisition_dict)):
            self.model.addConstr(list_exp[i] <= 1)

        # for each node we create the list of incoming and outgoing edges

        incoming_exp = []
        outgoing_exp = []
        for i in range(nb_nodes):
            incoming_exp.append(LinExpr())
            outgoing_exp.append(LinExpr())

        i = 0
        for edge in self.problem.graph.edges():
            e = incoming_exp[self.problem.id_to_ind[edge[1]]]
            e.add(vars_edge[i])
            f = outgoing_exp[self.problem.id_to_ind[edge[0]]]
            f.add(vars_edge[i])
            i += 1

        # flow constraint : same number of incoming and outgoing edges (max one)

        for node in self.problem.graph.nodes():
            if node["acquisition_request_id"] == "fisrt":
                self.model.addConstr(outgoing_exp[self.problem.id_to_ind[node]] == 1)
            elif node["acquisition_request_id"] == "final":
                self.model.addConstr(incoming_exp[self.problem.id_to_ind[node]] == 1)
            else:
                self.model.addConstr(incoming_exp[self.problem.id_to_ind[node]] == outgoing_exp[self.problem.id_to_ind[node]])
                # a node is chosen if at less one incoming edge is chosen
                # if a node is chosen, at less one outgoing edge is chosen
                self.model.addConstr(incoming_exp[self.problem.id_to_ind[node]] == vars_node[self.problem.id_to_ind[node]])
                self.model.addConstr(outgoing_exp[self.problem.id_to_ind[node]] == vars_node[self.problem.id_to_ind[node]])

