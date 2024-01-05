from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
import subprocess
from maximum_independent_set.solvers.mis_solvers import MisSolver
from maximum_independent_set.mis_problem import MisProblem, MisSolution
import os
this_dir = os.path.abspath(os.path.dirname(__file__))
# TODO : put this in config file
redumis = "/Users/poveda_g/Documents/git_repos/pub_repo/KaMIS/deploy/redumis"
online_mis = "/Users/poveda_g/Documents/git_repos/pub_repo/KaMIS/deploy/online_mis"
weighted = "/Users/poveda_g/Documents/git_repos/pub_repo/KaMIS/deploy/weighted_branch_reduce"
weighted_local_search = "/Users/poveda_g/Documents/git_repos/pub_repo/KaMIS/deploy/weighted_local_search"
methods = {"redumis": redumis,
           "weighted": weighted,
           "weighted_local_search": weighted_local_search,
           "online_mis": online_mis}
tmp_dir = os.path.join(this_dir, "tmp/")
import time
import logging
logger = logging.getLogger(__file__)
class MisKamisSolver(MisSolver):
    def init_model(self):
        t = time.time_ns()
        file = os.path.join(tmp_dir, f"file_{t}.graph")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        lines_to_write = []
        lines_to_write += [f"{self.problem.number_nodes} {len(self.problem.edges)} 10\n"]
        nb_edge = 0
        for i in range(self.problem.number_nodes):
            node = self.problem.index_to_nodes[i]
            neighs = list(self.problem.graph.neighbors(node))
            s = f"{int(self.problem.attr_list[i])} "+" ".join([f"{self.problem.nodes_to_index[x]+1}"
                                                          for x in neighs])
            nb_edge += len(neighs)
            lines_to_write += [s+"\n"]
        print(nb_edge, len(self.problem.edges))
        with open(file, "w") as f:
            f.writelines(lines_to_write)
        self.current_tag = t
        self.current_file = file

    def solve(self, **kwargs) -> ResultStorage:
        time_limit = kwargs.get("time_limit", 20)
        method = kwargs.get("method", "weighted_local_search")
        if method not in methods:
            logger.warning(f"{method} is not available, please use one of : {list(methods.keys())}")
            logger.warning("setting to default value weighted_local_search")
            method = "weighted_local_search"
        output = os.path.join(tmp_dir, f"res_{self.current_tag}.txt")
        command = (f"{methods[method]} {self.current_file} "
                   f"--output {output} --time_limit={time_limit} --console_log")
        # Launch the command and wait for it to finish
        result = subprocess.run(command, shell=True, check=True)
        chosen = []
        with open(output, "r") as file:
            for line in file:
                # Convert each line to an integer and append to the list
                number = int(line.strip())  # Remove leading/trailing whitespaces and convert to int
                chosen.append(number)
        sol = MisSolution(problem=self.problem, chosen=chosen)
        fit = self.aggreg_sol(sol)
        return ResultStorage([(sol, fit)],
                             mode_optim=self.params_objective_function.sense_function)









