import sys
import os

# Path Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)


from multibatching.solvers.cpsat import (CpsatMultibatchingSolver, 
                                            ModelingMultiBatch,
                                            CpsatBendersMultibatchingSolver)
from multibatching.solvers.packing_subproblem import (CpsatPackingSubproblem,
                                                        GreedyPackingForMultibatching)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from multibatching.solvers.cpmpy import CpmpyMultibatchingSolver
from multibatching.solvers.oe.solver_2 import OpenEvolveMultibatchingSolver
from multibatching.solvers.netx import NetxMultibatchingSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from multibatching.solvers.milp_flow import GurobiMultibatchingSolver
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from multibatching.solvers.column_generation import ColumnGenerationSolver
from multibatching.solvers.asp import ClingconMultibatchingSolver



def run_networkx(problem, p, time_limit, **kwargs):
    print("Starting NetworkX Solver...")
    solver = NetxMultibatchingSolver(problem)
    res_ = solver.solve()
    sol = res_[-1][0]
    subsolver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
    subsolver.init_from_solution(solution=sol)
    subsolver.init_model(solution=sol)
    return subsolver.solve(parameters_cp=p,
                           time_limit=time_limit,
                           ortools_cpsat_solver_kwargs={"log_search_progress": True})

def run_openevolve(problem, **kwargs):
    print("Starting OpenEvolve Solver...")
    solver = OpenEvolveMultibatchingSolver(problem)
    return solver.solve()

def run_column_generation(problem, **kwargs):
    solver = ColumnGenerationSolver(problem=problem)
    solver.init_model()
    return solver.solve() 

def run_asp(problem, time_limit, **kwargs):
    solver = ClingconMultibatchingSolver(problem)
    solver.init_model()
    return solver.solve(time_limit=time_limit)


def run_greedy_packing(problem, p, time_limit, **kwargs):
    print("Starting GreedyPacking strategy...")
    solver = CpsatMultibatchingSolver(problem=problem, scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW, single_batching=True)
    
    # Greedy specific overrides
    p.nb_process = 32
    
    print("Solving Phase 1 (CP-SAT)...")
    result = solver.solve(callbacks=[], time_limit=time_limit, parameters_cp=p, 
                          ortools_cpsat_solver_kwargs=dict(log_search_progress=True))
    
    if result:
        sol = result[-1][0]
        print("Phase 1 complete. Starting GreedyPackingForMultibatching...")
        greedy_ = GreedyPackingForMultibatching(problem=problem)
        greedy_.init_from_solution(sol)
        return greedy_.solve(single_batching=True)
    return None

def run_flow_postpro(problem, p, time_limit, **kwargs):
    print("Phase 1: Flow Model...")
    solver = CpsatMultibatchingSolver(problem, scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW, detailed_capacity_constraint=False)
    
    res1 = solver.solve(callbacks=[], parameters_cp=p, time_limit=time_limit/2, ortools_cpsat_solver_kwargs=kwargs)
    
    if res1:
        print("Phase 2: Packing Subproblem...")
        subsolver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
        # Using the best solution from Phase 1
        subsolver.init_from_solution(solution=res1[-1][0])
        subsolver.init_model(solution=res1[-1][0])
        return subsolver.solve(parameters_cp=p, time_limit=time_limit/10, ortools_cpsat_solver_kwargs=kwargs)
    return None

def run_flow_postpro_complete(problem, p, time_limit, **kwargs):
    print("Phase 1: Flow Model...")
    solver = CpsatMultibatchingSolver(problem, scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW, detailed_capacity_constraint=False)  
    res1 = solver.solve(callbacks=[], parameters_cp=p, time_limit=time_limit/2, ortools_cpsat_solver_kwargs=kwargs)

    if res1:
        print("Phase 2: Packing Subproblem...")
        subsolver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
        # Using the best solution from Phase 1
        subsolver.init_from_solution(solution=res1[-1][0])
        subsolver.init_model(solution=res1[-1][0])
        res2 = subsolver.solve(parameters_cp=p, time_limit=time_limit/10, ortools_cpsat_solver_kwargs=kwargs)

        if res2:
            print("Phase 3: Unit Flow Model...")
            sol = res2[-1][0]
            complete_solver = CpsatMultibatchingSolver(problem=problem)
            complete_solver.init_model(modeling=ModelingMultiBatch.UNIT_FLOW)
            complete_solver.set_warm_start(sol)
            result = complete_solver.solve(parameters_cp=p,
                                        time_limit=time_limit/2,
                                        ortools_cpsat_solver_kwargs={"log_search_progress": True})
            sol, _ = result.get_best_solution_fit()
            return sol
        return None
    return None

def run_flow_greedy(problem, p, time_limit, **kwargs):
    print("Phase 1: Flow Model...")
    solver = CpsatMultibatchingSolver(problem=problem,
                                      scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      single_batching=True)    
    p.nb_process = 32
    result = solver.solve(callbacks=[],
                          time_limit=time_limit,
                          parameters_cp=p,
                          ortools_cpsat_solver_kwargs=
                          dict(log_search_progress=True))
    sol = result.get_best_solution()

    if sol:
        print("Phase 2: Greedy Packing...")
        greedy_ = GreedyPackingForMultibatching(problem=problem)
        greedy_.init_from_solution(sol)

       
        return greedy_.solve(single_batching=True)
    return None

def run_benders_greedy(problem, p, time_limit, **kwargs):
    solver = CpsatBendersMultibatchingSolver(problem=problem,
                                             scaling_factor=1)
    solver.init_model(detailed_capacity_constraint=False,
                      single_batching=True)
    p = ParametersCp.default_cpsat()
    p.nb_process = 32
    subbrick = SubBrick(cls=GreedyPackingForMultibatching,
                        kwargs=dict(single_batching=True))
    result = solver.solve(callbacks=[],
                          nb_iteration=12,
                          params_solver_master=
                          {"time_limit": time_limit,
                           "parameters_cp": p,
                           "ortools_cpsat_solver_kwargs": {"log_search_progress": True}},
                          packing_solver=subbrick)
    sol, _ = result.get_best_solution_fit(problem)

    return [(sol, _)]

def run_milp_single(problem, p, time_limit, **kwargs):
    solver = GurobiMultibatchingSolver(problem)
    p_new = ParametersMilp.default()
    solver.init_model(single_batching=True)
    res = solver.solve(parameters_milp=p_new, time_limit=time_limit)
    sol = res.get_best_solution()
    return sol

def run_flow_model(problem, p, time_limit, **kwargs):
    print("Building Flow Model...")
    solver = CpsatMultibatchingSolver(problem)
    solver.init_model(modeling=ModelingMultiBatch.FLOW, detailed_capacity_constraint=True)
    return solver.solve(parameters_cp=p, time_limit=time_limit, ortools_cpsat_solver_kwargs={"log_search_progress": True})

def run_unit_model(problem, p, time_limit, **kwargs):
    print("Building Unit Flow Model...")
    solver = CpsatMultibatchingSolver(problem)
    solver.init_model(modeling=ModelingMultiBatch.UNIT_FLOW)
    return solver.solve(parameters_cp=p, time_limit=time_limit, ortools_cpsat_solver_kwargs=kwargs)


def run_benders(problem, p, time_limit, **kwargs):
    print("Configuring Benders Decomposition...")
    solver = CpsatBendersMultibatchingSolver(problem)
    solver.init_model(detailed_capacity_constraint=True)
    p.nb_process = 4 
    return solver.solve(
        params_solver_sub={'parameters_cp':p, 'time_limit': 5},
        params_solver_master={'parameters_cp':p, 'time_limit': time_limit},
        nb_iteration=30
    )

def run_cpmpy(problem, p, time_limit, **kwargs):
    print("Starting CPMpy Solver...")
    solver = CpmpyMultibatchingSolver(problem, scaling_factor=1, solver_name="exact")
    return solver.solve(parameters_cp=p, time_limit=time_limit, ortools_cpsat_solver_kwargs={"log_search_progress": True})