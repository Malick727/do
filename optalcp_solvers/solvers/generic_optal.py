from abc import abstractmethod

import json

import subprocess

from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
from discrete_optimization.generic_tools.cp_tools import CpSolver, ParametersCp
from discrete_optimization.generic_tools.do_problem import (Problem,
                                                            Solution,
                                                            ParamsObjectiveFunction)
from typing import Optional, Any
import pandas as pd
import os

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

this_folder = os.path.abspath(os.path.dirname(__file__))


class OptalBasicCallback(BasicStatsCallback):
    def __init__(self, stats: dict):
        self.brut_stats = stats

    def get_df_metrics(self) -> pd.DataFrame:
        """Construct a dataframe indexed by time of the recorded metrics (fitness, bounds...)."""
        # column_names = [self.time_column] + self.metric_columns
        # df = pd.DataFrame(
        #    [{k: v for k, v in st.items() if k in column_names} for st in self.stats]
        # ).set_index(self.time_column)
        # df.columns.name = "metric"
        obj_hist = [x["objective"] for x in self.brut_stats["objectiveHistory"]]
        time_obj = [x["solveTime"] for x in self.brut_stats["objectiveHistory"]]
        bound_hist = [x["value"] for x in self.brut_stats["lowerBoundHistory"]]
        time_bound = [x["solveTime"] for x in self.brut_stats["lowerBoundHistory"]]
        records = []
        for i in range(len(time_obj)):
            t = time_obj[i]
            index_on_bound = next((j for j in range(len(time_bound)-1)
                                   if time_bound[j] <= t <= time_bound[j + 1]), None)
            if index_on_bound is None:
                index_on_bound = len(time_bound)-1
            d = {"time": t, "obj": obj_hist[i], "fit": obj_hist[i], "bound": bound_hist[index_on_bound]}
            records.append(d)
        df = pd.DataFrame.from_records(records)
        df.set_index("time", inplace=True)

        return df


class OptalSolver(CpSolver):
    def __init__(self, problem: Problem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.temp_directory = os.path.join(this_folder, "temp/")
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory)
        self._file_input: str = None
        self._logs_path: str = None
        self._result_path: str = None
        self._stats: dict = None
        self._script_model: str = None
        self._is_init: bool = False

    def build_command(self,
                      parameters_cp: Optional[ParametersCp] = None,
                      time_limit: int = 10,
                      **args: Any):
        ARGUMENTS = [f"--timeLimit {time_limit}",
                     f"--nbWorkers {parameters_cp.nb_process}"]
        for k in args:
            ARGUMENTS.append(f"--{k} {args[k]}")
        # ARGUMENTS.append(f"--result {self._logs_path}")
        command = [
            "node",
            self._script_model,
            self._file_input,
            *ARGUMENTS
        ]
        return command

    @abstractmethod
    def retrieve_current_solution(self, dict_results: dict)->Solution:
        ...

    def solve(self, callbacks: Optional[list[Callback]] = None,
              parameters_cp: Optional[ParametersCp] = None,
              time_limit: int = 10,
              **args: Any) -> ResultStorage:
        if not self._is_init:
            self.init_model(**args)
            self._is_init = True
        cb_list = CallbackList(callbacks=callbacks)
        cb_list.on_solve_start(self)
        command = self.build_command(parameters_cp=parameters_cp,
                                     time_limit=time_limit,
                                     **args)

        try:
            str_command = " ".join(command)
            res_command = os.system(str_command)
            dict_ = json.load(open(self._result_path, "r"))
            sol = self.retrieve_current_solution(dict_results=dict_)
            fit = self.aggreg_from_sol(sol)
            # self._stats = json.load(open(self._logs_path, "r"))
            self._stats = dict_
            if self._stats["lowerBoundHistory"][-1]["value"] == self._stats["objectiveHistory"][-1]["objective"]:
                self.status_solver = StatusSolver.OPTIMAL
            else:
                self.status_solver = StatusSolver.SATISFIED
            res = self.create_result_storage(list_solution_fits=[(sol, fit)])
            cb_list.on_solve_end(res=res, solver=self)
            return res
        except FileNotFoundError:
            print(str_command)
            print(f"Error: The command 'node' or script '{self._script_model}' was not found.")
            print("Please ensure Node.js is installed and the path to your script is correct.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error: Command failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return None


    def get_output_stats(self):
        return self._stats