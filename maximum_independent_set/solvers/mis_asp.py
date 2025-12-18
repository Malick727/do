#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import time
from typing import Any, List, Optional

import clingo
from clingo import Symbol

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver_with_starting_solution import (
    SolverColoringWithStartingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from maximum_independent_set.mis_problem import MisProblem, MisSolution
from maximum_independent_set.solvers.mis_solvers import MisSolver
cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class MisASPSolver(MisSolver):
    def init_model(self, **kwargs: Any) -> None:
        if self.problem.attribute_aggregate == "size":
            basic_model = """
            {independent_set(V):vertex(V)}.
            independent_set(V) :- vertex(V), not {independent_set(U) : edge(U, V)}.
            :- edge(X,Y), independent_set(X), independent_set(Y).
            #maximize {1, V:independent_set(V)}.
            #show independent_set/1.
            """
        else:
            basic_model = """
                        {independent_set(V):vertex(V)}.
                        independent_set(V) :- vertex(V), not {independent_set(U) : edge(U, V)}.
                        :- edge(X,Y), independent_set(X), independent_set(Y).
                        #maximize {W,V : independent_set(V), weight(V,W)}.
                        #show independent_set/1.
                        """
        max_models = kwargs.get("max_models", 100)
        self.ctl = clingo.Control(
            ["--warn=no-atom-undefined", f"--models={max_models}", "--opt-mode=opt", f"--models={max_models}"]
        )
        self.ctl.add("base", [], basic_model)
        string_data_input = self.build_data_string()
        self.ctl.add("base", [], string_data_input)

    def build_data_string(self):
        s = f"vertex(0..{self.problem.number_nodes-1}).\n"
        s += "% Define the edges in the graph. Modify this according to your graph structure.\n"
        for edge in self.problem.edges:
            s += f"edge({self.problem.nodes_to_index[edge[0]]},{self.problem.nodes_to_index[edge[1]]}).\n"
        if self.problem.attribute_aggregate != "size":
            for i in range(self.problem.number_nodes):
                s += f"weight({i}, {int(self.problem.attr_list[i])}).\n"
        #s += f"independent_set(0).\n"
        #s += f"independent_set(1).\n"

        return s

    def solve(self, **kwargs: Any) -> ResultStorage:
        start_time_grounding = time.perf_counter()
        self.ctl.ground([("base", [])])
        logger.info(
            f"Grounding programs: ...\n=== Grounding done"
            f" {time.perf_counter() - start_time_grounding} sec ==="
        )

        class CallbackASP:
            def __init__(
                    self,
                    dump_model_in_folders: bool = False,
            ):
                self.nb_found_models = 0
                self.current_time = time.perf_counter()
                self.model_results = []
                self.symbols_results = []
                self.dump_model_in_folders = dump_model_in_folders

            def on_model(self, m: clingo.Model):
                self.model_results += [m]
                self.symbols_results += [m.symbols(atoms=True)]
                self.nb_found_models += 1
                logger.info(
                    f"=== New Model [{self.nb_found_models}]"
                    f" found after {time.perf_counter() - self.current_time}"
                    f" sec of solving === "
                )
                logger.info(f"=== cost = {m.cost} ===")
                logger.info(f"=== Optimality proven ? {m.optimality_proven} === ")
                if m.optimality_proven:
                    return 0
                if self.dump_model_in_folders:
                    folder_model = os.path.join(
                        cur_folder, f"output-folder/model_{self.nb_found_models}"
                    )
                    create_empty_folder(folder_model)
                    logger.info("Dumping model.txt ...")
                    with open(
                            os.path.join(folder_model, "model.txt"), "w"
                    ) as model_file:
                        model_file.write(str(m))

        timeout_seconds = kwargs.get("timeout_seconds", 100)
        callback = CallbackASP(
            dump_model_in_folders=kwargs.get("dump_model_in_folders", False),
        )
        with self.ctl.solve(on_model=callback.on_model, async_=True) as handle:
            handle.wait(timeout_seconds)
            handle.cancel()
        return self.retrieve_solutions([callback.symbols_results[-1]])

    def retrieve_solutions(self, symbols_results):
        list_solutions_fit = []
        for symbols in symbols_results:
            print(len(symbols))
            set_is = [
                s.arguments[0].number
                for s in symbols
                if s.name == "independent_set"
            ]
            print(set_is)
            empty_sol = [0 for i in range(self.problem.number_nodes)]
            for s in set_is:
                empty_sol[s] = 1
            solution = MisSolution(problem=self.problem, chosen=empty_sol)
            fit = self.aggreg_sol(solution)
            list_solutions_fit += [(solution, fit)]
        return ResultStorage(list_solution_fits=list_solutions_fit,
                             mode_optim=self.params_objective_function.sense_function)

def create_empty_folder(folder):
    logger.info(f"Creating empty folder: {folder}")
    if os.path.exists(folder):
        os.removedirs(folder)
    os.makedirs(folder)
