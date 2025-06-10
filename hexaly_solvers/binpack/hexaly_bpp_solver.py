import math
from typing import Any

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.do_problem import Solution
from hexaly_solvers.hexaly_solver import HexalySolver
import hexaly.optimizer

class HexalyBinPackSolver(HexalySolver):
    problem: BinPackProblem
    def init_model(self, **kwargs: Any) -> None:
        """
        Mostly adaptation
        from https://www.hexaly.com/example/bin-packing-problem-with-conflicts-bppc"""
        self.variables = {}
        weights = [self.problem.list_items[i].weight for i in range(self.problem.nb_items)]
        optimizer = hexaly.optimizer.HexalyOptimizer()
        nb_min_bins = int(math.ceil(sum(weights) / float(self.problem.capacity_bin)))
        nb_max_bins = min(self.problem.nb_items, 2 * nb_min_bins)

        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Set decisions: bins[k] represents the items in bin k
        bins = [model.set(self.problem.nb_items) for _ in range(nb_max_bins)]

        # Transform bins and itemFordbidden list into hx expression
        neighs_dict = {}
        for i in range(self.problem.nb_items):
            edges = [x for x in self.problem.incompatible_items if x[0] == i or x[1] == i]
            neighs = [x[0] if x[0] != i else x[1] for x in edges]
            neighs_dict[i] = neighs
        bins_array = model.array(bins)
        forbidden_items_array = model.array([neighs_dict[i] for i in range(self.problem.nb_items)])
        # Find the bin where an item is packed
        bin_for_item = [model.find(bins_array, i) for i in range(self.problem.nb_items)]
        # Each item must be in one bin and one bin only
        model.constraint(model.partition(bins))

        # Create an array and a function to retrieve the item's weight
        weights = model.array(weights)
        weight_lambda = model.lambda_function(lambda i: weights[i])

        # Forbidden constraint for each items
        for i in range(self.problem.nb_items):
            items_intersection = model.intersection(forbidden_items_array[i], bins_array[bin_for_item[i]])
            model.constraint(model.count(items_intersection) == 0)

        # Weight constraint for each bin
        bin_weights = [model.sum(b, weight_lambda) for b in bins]
        for w in bin_weights:
            model.constraint(w <= self.problem.capacity_bin)

        # Bin k is used if at least one item is in it
        bins_used = [model.count(b) > 0 for b in bins]

        # Count the used bins
        total_bins_used = model.sum(bins_used)
        self.variables["bin_for_item"] = bin_for_item
        # Minimize the number of used bins
        model.minimize(total_bins_used)
        self.ls = optimizer
        model.close()

    def retrieve_solution(self, **args) -> Solution:
        objs = tuple([self.ls.model.objectives[i].value for i in range(self.ls.model.nb_objectives)])
        print("Objs =", objs)
        allocation = [self.variables["bin_for_item"][i].value for i in range(self.problem.nb_items)]
        return BinPackSolution(problem=self.problem, allocation=allocation)