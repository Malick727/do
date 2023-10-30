import logging
import random
from typing import Any, Dict, List, Set, Tuple, Optional

import networkx as nx
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lp_tools import (
    CplexMilpSolver,
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    ParametersMilp,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.pickup_vrp.solver.lp_solver import build_the_cycles
from gurobipy import GRB, Model, Var, quicksum

from path_planning.path_planning import (
    Comparison,
    NodeConstraints,
    PathPlanning,
    PathSolution, operator_map,
)

logger = logging.getLogger(__name__)

map_to_grb = {
    Comparison.UEQ: GRB.GREATER_EQUAL,
    Comparison.UP: GRB.GREATER_EQUAL,
    Comparison.LE: GRB.LESS_EQUAL,
    Comparison.LEQ: GRB.LESS_EQUAL,
    Comparison.EQ: GRB.EQUAL,
}


class LPPathPlanning_Gurobi(GurobiMilpSolver):
    def __init__(
        self,
        problem: PathPlanning,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
        self.model = None
        self.variables: Dict[str, "Var"] = {}

    def compute_edges_in_and_out(self):
        graph: nx.DiGraph = self.problem.graph.to_networkx()
        edges = set(graph.edges())
        edges_index = set(
            [
                (self.problem.index_nodes[e[0]], self.problem.index_nodes[e[1]])
                for e in edges
            ]
        )
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
        include_constraints = kwargs.get("include_constraints", True)
        edges_out, edges_in, edges_index = self.compute_edges_in_and_out()
        model = Model("path-planning")
        x = {
            e: model.addVar(
                obj=self.problem.matrix_length[e[0], e[1]],
                lb=0,
                ub=1,
                vtype=GRB.BINARY,
                name=f"x_{e}",
            )
            for e in edges_index
        }
        for node in self.problem.index_to_nodes:
            if node == self.problem.index_source:
                model.addLConstr(
                    quicksum([x[e] for e in edges_out[node]]) == 1,
                    name=f"outflow_{node}",
                )  # Outbound flow only.
                model.addLConstr(
                    quicksum([x[e] for e in edges_in.get(node, [])]) == 0,
                    name=f"inflow_{node}",
                )  # Outbound flow only.
                continue
            if node == self.problem.index_sink:
                model.addLConstr(
                    quicksum([x[e] for e in edges_out.get(node, [])]) == 0,
                    name=f"outflow_{node}",
                )  # No outbound
                model.addLConstr(
                    quicksum([x[e] for e in edges_in.get(node, [])]) == 1,
                    name=f"inflow_{node}",
                )  # Inbound flow only.
                continue
            model.addLConstr(
                quicksum(
                    [x[e] for e in edges_in.get(node, [])]
                    + [-x[e] for e in edges_out.get(node, [])]
                )
                == 0,
                name=f"flow_conv_{node}",
            )
            model.addLConstr(
                quicksum([x[e] for e in edges_in.get(node, [])]) <= 1,
                name=f"max_flowin_conv_{node}",
            )
            model.addLConstr(
                quicksum([x[e] for e in edges_out.get(node, [])]) <= 1,
                name=f"max_flowout_conv_{node}",
            )
        self.variables["x"] = x
        if include_constraints and not self.problem.node_constraints.is_empty():
            visit = {
                node: model.addVar(obj=0, vtype=GRB.BINARY, name=f"visit_{node}")
                for node in self.problem.node_constraints.node_to_index_constraint
            }
            self.variables["visit"] = visit
            for node in visit:
                if node in {self.problem.index_sink, self.problem.index_source}:
                    model.addLConstr(visit[node] == 1)
                    continue
                else:
                    if node in edges_in:
                        model.addLConstr(
                            quicksum([x[e] for e in edges_in[node]]) == visit[node]
                        )
            for j in range(self.problem.node_constraints.nb_constraints):
                c: Tuple[
                    List[int], Comparison, int
                ] = self.problem.node_constraints.list_constraint[j]
                bound = c[2]

                if c[1] == Comparison.UP:
                    bound = c[2] + 1
                if c[1] == Comparison.LE:
                    bound = c[2] - 1
                model.addLConstr(
                    quicksum([visit[n] for n in c[0]]),
                    sense=map_to_grb[c[1]],
                    rhs=bound,
                    name=f"node_constraint_{j}",
                )

        if include_constraints and not self.problem.path_constraints.is_empty():
            for j in range(self.problem.path_constraints.nb_constraints):
                c: Tuple[
                    str, Comparison, int
                ] = self.problem.path_constraints.list_constraint[j]
                bound = c[2]
                # TODO : dirty hardcoded things.
                if c[1] == Comparison.UP:
                    bound = c[2] + 0.00001
                if c[1] == Comparison.LE:
                    bound = c[2] - 0.00001
                model.addLConstr(
                    quicksum(
                        [
                            x[e] * self.problem.matrix_length_attr[c[0]][e[0], e[1]]
                            for e in x
                        ]
                    ),
                    sense=map_to_grb[c[1]],
                    rhs=bound,
                    name=f"path_constraint_{j}",
                )
        self.model = model

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        list_solution_fits = []
        nb_solution = self.nb_solutions
        for j in range(nb_solution):
            logger.info(
                f"{j}-th solution has a cost of {self.get_obj_value_for_ith_solution(j)}"
            )
            edges_selected = set()
            for e in self.variables["x"]:
                val = self.get_var_value_for_ith_solution(self.variables["x"][e], j)
                if val > 0.5:
                    edges_selected.add(e)
            _, nodes_path = self.rebuild_path(set_edges_selected=edges_selected)
            sol = PathSolution(problem=self.problem, path=nodes_path)
            fit = self.aggreg_sol(sol)
            list_solution_fits += [(sol, fit)]
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )

    def compute_graph_and_component(self, set_edges_selected: Set[Tuple[int, int]]):
        g = nx.DiGraph()
        g.add_edges_from(list(set_edges_selected))
        connected_components = [(cyc, len(cyc)) for cyc in nx.simple_cycles(g)]
        res = sorted(connected_components, key=lambda x: x[1], reverse=False)
        return g, res

    def rebuild_path(
        self, set_edges_selected: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        edg_copy = set(set_edges_selected)
        edges_path = []
        nodes_path = [self.problem.index_source]
        while len(edg_copy) > 0:
            next_edge = [e for e in edg_copy if e[0] == nodes_path[-1]]
            if len(next_edge) > 0:
                next_edge = next_edge[0]
            else:
                logger.debug(nodes_path, "Problem in rebuild_path ! ")
            edges_path += [next_edge]
            nodes_path += [next_edge[1]]
            edg_copy.remove(next_edge)
        return edges_path, nodes_path

        # def callback(model: grb.Model, where: int) -> None:
        #     if where == grb.GRB.Callback.MIPSOL:
        #         if self.problem.graph is None:
        #             raise RuntimeError(
        #                 "self.problem.graph cannot be None at this point"
        #             )
        #         solution = model.cbGetSolution(
        #             [
        #                 self.variable_decisions["variables_edges"][x[0]][x[1]]
        #                 for x in indexes_edges
        #             ]
        #         )
        #         flow_solution: Dict[int, Dict[int, int]] = {
        #             v: {} for v in range(self.problem.number_vehicle)
        #         }
        #         cost = {v: 0 for v in range(self.problem.number_vehicle)}
        #         for k in range(len(solution)):
        #             if solution[k] > 0.5:
        #                 flow_solution[indexes_edges[k][0]][indexes_edges[k][1]] = 1
        #                 cost[
        #                     indexes_edges[k][0]
        #                 ] += self.problem.graph.edges_infos_dict[indexes_edges[k][1]][
        #                     "distance"
        #                 ]
        #         list_temporary_results = build_graph_solutions(
        #             solutions=[({"variables_edges": flow_solution}, 0)],
        #             graph=self.problem.graph,
        #         )
        #         temporary_result = list_temporary_results[0]
        #         reevaluate_result(temporary_result, problem=self.problem)
        #         vehicle_keys = temporary_result.graph_vehicle.keys()
        #         connected_components = {
        #             v: [
        #                 (cyc, len(cyc))
        #                 for cyc in nx.simple_cycles(temporary_result.graph_vehicle[v])
        #             ]
        #             for v in vehicle_keys
        #         }
        #         logger.debug(cost)
        #         logger.debug(
        #             f"Cycles : {[len(connected_components[v]) for v in connected_components]}"
        #         )
        #         sorted_connected_component = {
        #             v: sorted(
        #                 connected_components[v], key=lambda x: x[1], reverse=False
        #             )
        #             for v in connected_components
        #         }
        #         temporary_result.connected_components_per_vehicle = (
        #             sorted_connected_component
        #         )
        #
        #         if temporary_result.paths_component is not None:
        #             for v in temporary_result.paths_component:
        #                 if len(temporary_result.paths_component[v]) > 1:
        #                     for p_ind in temporary_result.paths_component[v]:
        #                         pp = temporary_result.paths_component[v][p_ind] + [
        #                             temporary_result.paths_component[v][p_ind][0]
        #                         ]
        #                         if self.problem.target_vehicle[v] not in pp:
        #                             for vv in temporary_result.paths_component:
        #                                 if all(
        #                                     (e0, e1)
        #                                     in self.variable_decisions[
        #                                         "variables_edges"
        #                                     ][vv]
        #                                     for e0, e1 in zip(pp[:-1], pp[1:])
        #                                 ):
        #                                     keys = [
        #                                         (e0, e1)
        #                                         for e0, e1 in zip(pp[:-1], pp[1:])
        #                                     ]
        #                                     model.cbLazy(
        #                                         grb.quicksum(
        #                                             [
        #                                                 self.variable_decisions[
        #                                                     "variables_edges"
        #                                                 ][vv][key]
        #                                                 for key in keys
        #                                             ]
        #                                         )
        #                                         <= len(pp) - 2
        #                                     )
        #         for v in sorted_connected_component:
        #             if len(sorted_connected_component[v]) > 1:
        #                 for s, l in sorted_connected_component[v]:
        #                     keys = [(e0, e1) for e0, e1 in zip(s[:-1], s[1:])] + [
        #                         (s[-1], s[0])
        #                     ]
        #                     model.cbLazy(
        #                         grb.quicksum(
        #                             [
        #                                 self.variable_decisions["variables_edges"][v][
        #                                     key
        #                                 ]
        #                                 for key in keys
        #                             ]
        #                         )
        #                         <= len(keys) - 1
        #                     )
        #
        #         subtour: Union[SubtourAddingConstraint, SubtourAddingConstraintCluster]
        #         if not self.clusters_version:
        #             subtour = SubtourAddingConstraint(
        #                 problem=self.problem, linear_solver=self, lazy=True
        #             )
        #             subtour.adding_component_constraints([temporary_result])
        #         else:
        #             subtour = SubtourAddingConstraintCluster(
        #                 problem=self.problem, linear_solver=self, lazy=True
        #             )
        #             subtour.adding_component_constraints([temporary_result])


class LPPathPlanning_GurobiLazyConstraint(GurobiMilpSolver):
    def __init__(
        self,
        problem: PathPlanning,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
        self.model = None
        self.variables: Dict[str, "Var"] = {}

    def compute_edges_in_and_out(self):
        graph: nx.DiGraph = self.problem.graph.to_networkx()
        edges = set(graph.edges())
        edges_index = set(
            [
                (self.problem.index_nodes[e[0]], self.problem.index_nodes[e[1]])
                for e in edges
            ]
        )
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
        include_constraints = kwargs.get("include_constraints", True)
        edges_out, edges_in, edges_index = self.compute_edges_in_and_out()
        model = Model("path-planning")
        x = {
            e: model.addVar(
                obj=self.problem.matrix_length[e[0], e[1]],
                lb=0,
                ub=1,
                vtype=GRB.BINARY,
                name=f"x_{e}",
            )
            for e in edges_index
        }
        for node in self.problem.index_to_nodes:
            if node == self.problem.index_source:
                model.addLConstr(
                    quicksum([x[e] for e in edges_out[node]]) == 1,
                    name=f"outflow_{node}",
                )  # Outbound flow only.
                model.addLConstr(
                    quicksum([x[e] for e in edges_in.get(node, [])]) == 0,
                    name=f"inflow_{node}",
                )  # Outbound flow only.
                continue
            if node == self.problem.index_sink:
                model.addLConstr(
                    quicksum([x[e] for e in edges_out.get(node, [])]) == 0,
                    name=f"outflow_{node}",
                )  # No outbound
                model.addLConstr(
                    quicksum([x[e] for e in edges_in.get(node, [])]) == 1,
                    name=f"inflow_{node}",
                )  # Inbound flow only.
                continue
            model.addLConstr(
                quicksum(
                    [x[e] for e in edges_in.get(node, [])]
                    + [-x[e] for e in edges_out.get(node, [])]
                )
                == 0,
                name=f"flow_conv_{node}",
            )
            model.addLConstr(
                quicksum([x[e] for e in edges_in.get(node, [])]) <= 1,
                name=f"max_flowin_conv_{node}",
            )
            model.addLConstr(
                quicksum([x[e] for e in edges_out.get(node, [])]) <= 1,
                name=f"max_flowout_conv_{node}",
            )
        self.variables["x"] = x
        if include_constraints and not self.problem.node_constraints.is_empty():
            visit = {
                node: model.addVar(obj=0, vtype=GRB.BINARY, name=f"visit_{node}")
                for node in self.problem.node_constraints.node_to_index_constraint
            }
            self.variables["visit"] = visit
            for node in visit:
                if node in {self.problem.index_sink, self.problem.index_source}:
                    model.addLConstr(visit[node] == 1)
                    continue
                else:
                    if node in edges_in:
                        model.addLConstr(
                            quicksum([x[e] for e in edges_in[node]]) == visit[node]
                        )
            for j in range(self.problem.node_constraints.nb_constraints):

                c: Tuple[
                    List[int], Comparison, int
                ] = self.problem.node_constraints.list_constraint[j]
                if random.random() < 0.5 and len(c[0]) < 10:
                    continue
                bound = c[2]
                if c[1] == Comparison.UP:
                    bound = c[2] + 1
                if c[1] == Comparison.LE:
                    bound = c[2] - 1
                model.addLConstr(
                    quicksum([visit[n] for n in c[0]]),
                    sense=map_to_grb[c[1]],
                    rhs=bound,
                    name=f"node_constraint_{j}",
                )
        if include_constraints and not self.problem.path_constraints.is_empty():
            for j in range(self.problem.path_constraints.nb_constraints):
                c: Tuple[
                    str, Comparison, int
                ] = self.problem.path_constraints.list_constraint[j]
                bound = c[2]
                # TODO : dirty hardcoded things.
                if c[1] == Comparison.UP:
                    bound = c[2] + 0.00001
                if c[1] == Comparison.LE:
                    bound = c[2] - 0.00001
                model.addLConstr(
                    quicksum(
                        [
                            x[e] * self.problem.matrix_length_attr[c[0]][e[0], e[1]]
                            for e in x
                        ]
                    ),
                    sense=map_to_grb[c[1]],
                    rhs=bound,
                    name=f"path_constraint_{j}",
                )
        self.model = model
        self.model.Params.LazyConstraints = 1

    def optimize_model(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> None:
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.prepare_model(parameters_milp=parameters_milp, **kwargs)

        def lazyconstr(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(self.variables["x"])
                edges_ = {x for x in vals if vals[x] > 0.5}
                print("Obj : ", model.cbGet(GRB.Callback.MIPSOL_OBJBND))
                for j in range(self.problem.node_constraints.nb_constraints):
                    c: Tuple[
                        List[int], Comparison, int
                    ] = self.problem.node_constraints.list_constraint[j]
                    edges_c = [x for x in edges_ if x[1] in c[0]]
                    nb_visit = len(edges_c)
                    bound = c[2]
                    constr = self.problem.node_constraints.list_constraint[j]
                    if not operator_map[constr[1]](nb_visit, bound):
                        print("nodes : ", c[0])
                        print("nb visit : ", nb_visit, c[2])
                        print(edges_c)
                        #model.cbLazy(quicksum(self.variables["x"][y] for y in edges_c) <= bound)
                        model.cbLazy(
                            quicksum([self.variables["visit"][n] for n in c[0]]),
                            sense=map_to_grb[c[1]],
                            rhs=bound,
                        )
        self.model.optimize(lazyconstr)
        logger.info(f"Problem has {self.model.NumObj} objectives")
        logger.info(f"Solver found {self.model.SolCount} solutions")
        logger.info(f"Objective : {self.model.getObjective().getValue()}")

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        list_solution_fits = []
        nb_solution = self.nb_solutions
        for j in range(nb_solution):
            logger.info(
                f"{j}-th solution has a cost of {self.get_obj_value_for_ith_solution(j)}"
            )
            edges_selected = set()
            for e in self.variables["x"]:
                val = self.get_var_value_for_ith_solution(self.variables["x"][e], j)
                if val > 0.5:
                    edges_selected.add(e)
            _, nodes_path = self.rebuild_path(set_edges_selected=edges_selected)
            sol = PathSolution(problem=self.problem, path=nodes_path)
            fit = self.aggreg_sol(sol)
            list_solution_fits += [(sol, fit)]
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )

    def compute_graph_and_component(self, set_edges_selected: Set[Tuple[int, int]]):
        g = nx.DiGraph()
        g.add_edges_from(list(set_edges_selected))
        connected_components = [(cyc, len(cyc)) for cyc in nx.simple_cycles(g)]
        res = sorted(connected_components, key=lambda x: x[1], reverse=False)
        return g, res

    def rebuild_path(
        self, set_edges_selected: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        edg_copy = set(set_edges_selected)
        edges_path = []
        nodes_path = [self.problem.index_source]
        while len(edg_copy) > 0:
            next_edge = [e for e in edg_copy if e[0] == nodes_path[-1]]
            if len(next_edge) > 0:
                next_edge = next_edge[0]
            else:
                logger.debug(nodes_path, "Problem in rebuild_path ! ")
            edges_path += [next_edge]
            nodes_path += [next_edge[1]]
            edg_copy.remove(next_edge)
        return edges_path, nodes_path

        # def callback(model: grb.Model, where: int) -> None:
        #     if where == grb.GRB.Callback.MIPSOL:
        #         if self.problem.graph is None:
        #             raise RuntimeError(
        #                 "self.problem.graph cannot be None at this point"
        #             )
        #         solution = model.cbGetSolution(
        #             [
        #                 self.variable_decisions["variables_edges"][x[0]][x[1]]
        #                 for x in indexes_edges
        #             ]
        #         )
        #         flow_solution: Dict[int, Dict[int, int]] = {
        #             v: {} for v in range(self.problem.number_vehicle)
        #         }
        #         cost = {v: 0 for v in range(self.problem.number_vehicle)}
        #         for k in range(len(solution)):
        #             if solution[k] > 0.5:
        #                 flow_solution[indexes_edges[k][0]][indexes_edges[k][1]] = 1
        #                 cost[
        #                     indexes_edges[k][0]
        #                 ] += self.problem.graph.edges_infos_dict[indexes_edges[k][1]][
        #                     "distance"
        #                 ]
        #         list_temporary_results = build_graph_solutions(
        #             solutions=[({"variables_edges": flow_solution}, 0)],
        #             graph=self.problem.graph,
        #         )
        #         temporary_result = list_temporary_results[0]
        #         reevaluate_result(temporary_result, problem=self.problem)
        #         vehicle_keys = temporary_result.graph_vehicle.keys()
        #         connected_components = {
        #             v: [
        #                 (cyc, len(cyc))
        #                 for cyc in nx.simple_cycles(temporary_result.graph_vehicle[v])
        #             ]
        #             for v in vehicle_keys
        #         }
        #         logger.debug(cost)
        #         logger.debug(
        #             f"Cycles : {[len(connected_components[v]) for v in connected_components]}"
        #         )
        #         sorted_connected_component = {
        #             v: sorted(
        #                 connected_components[v], key=lambda x: x[1], reverse=False
        #             )
        #             for v in connected_components
        #         }
        #         temporary_result.connected_components_per_vehicle = (
        #             sorted_connected_component
        #         )
        #
        #         if temporary_result.paths_component is not None:
        #             for v in temporary_result.paths_component:
        #                 if len(temporary_result.paths_component[v]) > 1:
        #                     for p_ind in temporary_result.paths_component[v]:
        #                         pp = temporary_result.paths_component[v][p_ind] + [
        #                             temporary_result.paths_component[v][p_ind][0]
        #                         ]
        #                         if self.problem.target_vehicle[v] not in pp:
        #                             for vv in temporary_result.paths_component:
        #                                 if all(
        #                                     (e0, e1)
        #                                     in self.variable_decisions[
        #                                         "variables_edges"
        #                                     ][vv]
        #                                     for e0, e1 in zip(pp[:-1], pp[1:])
        #                                 ):
        #                                     keys = [
        #                                         (e0, e1)
        #                                         for e0, e1 in zip(pp[:-1], pp[1:])
        #                                     ]
        #                                     model.cbLazy(
        #                                         grb.quicksum(
        #                                             [
        #                                                 self.variable_decisions[
        #                                                     "variables_edges"
        #                                                 ][vv][key]
        #                                                 for key in keys
        #                                             ]
        #                                         )
        #                                         <= len(pp) - 2
        #                                     )
        #         for v in sorted_connected_component:
        #             if len(sorted_connected_component[v]) > 1:
        #                 for s, l in sorted_connected_component[v]:
        #                     keys = [(e0, e1) for e0, e1 in zip(s[:-1], s[1:])] + [
        #                         (s[-1], s[0])
        #                     ]
        #                     model.cbLazy(
        #                         grb.quicksum(
        #                             [
        #                                 self.variable_decisions["variables_edges"][v][
        #                                     key
        #                                 ]
        #                                 for key in keys
        #                             ]
        #                         )
        #                         <= len(keys) - 1
        #                     )
        #
        #         subtour: Union[SubtourAddingConstraint, SubtourAddingConstraintCluster]
        #         if not self.clusters_version:
        #             subtour = SubtourAddingConstraint(
        #                 problem=self.problem, linear_solver=self, lazy=True
        #             )
        #             subtour.adding_component_constraints([temporary_result])
        #         else:
        #             subtour = SubtourAddingConstraintCluster(
        #                 problem=self.problem, linear_solver=self, lazy=True
        #             )
        #             subtour.adding_component_constraints([temporary_result])
