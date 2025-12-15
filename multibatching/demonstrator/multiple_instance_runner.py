import sys
import os
import time
import random
from typing import Tuple, List, Callable, Any, Dict

import pandas as pd

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)

# --- PROJECT IMPORTS ---
from multibatching.problem import MultibatchingProblem
from multibatching.utils import generate_multibatching_problem
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.study import (
    Experiment,
    Hdf5Database,
    SolverConfig,
)

# --- RUNNERS IMPORT ---
# Attempts to import the 'runners' module which contains the solver execution logic.
try:
    from runners import (
        run_networkx,
        run_openevolve,
        run_greedy_packing,
        run_flow_postpro,
        # Add other runners here as needed
    )
except ImportError:
    # Fallback for different execution contexts (e.g., running from root)
    try:
        from multibatching.demonstrator.runners import (
            run_networkx,
            run_openevolve,
            run_greedy_packing,
            run_flow_postpro
        )
    except ImportError as e:
        print("CRITICAL ERROR: Could not import 'runners' module.")
        print("Ensure 'runners.py' is located in the same directory or available in the PYTHONPATH.")
        raise e

# ==============================================================================
# 1. DATASET GENERATION
# ==============================================================================

def generate_stratified_dataset(
    n_instances: int,
    location_intervals: List[Tuple[int, int]],
    transport_intervals: List[Tuple[int, int]],
    product_intervals: List[Tuple[int, int]],
    sparsity_intervals: List[Tuple[float, float]],
    master_seed: int = None,
    **fixed_kwargs
) -> List[MultibatchingProblem]:
    """
    Generates a dataset of MultibatchingProblem instances using stratified sampling.
    
    This function selects parameter ranges (intervals) randomly for each instance 
    to ensure a diverse dataset covering different complexity clusters (e.g., Small, Medium, Large).

    Args:
        n_instances: Total number of instances to generate.
        location_intervals: List of (min, max) tuples for the number of locations.
        transport_intervals: List of (min, max) tuples for the number of transport types.
        product_intervals: List of (min, max) tuples for the number of products.
        sparsity_intervals: List of (min, max) tuples for demand sparsity.
        master_seed: Optional seed for reproducibility.
        **fixed_kwargs: Static arguments passed directly to the generator 
                        (e.g., min_dist, max_tt_cost).

    Returns:
        List[MultibatchingProblem]: The generated dataset.
    """
    
    dataset = []
    if master_seed is not None:
        random.seed(master_seed)

    print(f"Starting stratified generation of {n_instances} instances...")

    for i in range(n_instances):
        try:
            # 1. Select random intervals for this specific instance
            loc_range = random.choice(location_intervals)
            trans_range = random.choice(transport_intervals)
            prod_range = random.choice(product_intervals)
            spars_range = random.choice(sparsity_intervals)

            # 2. Sample concrete values from the selected intervals
            n_loc = random.randint(loc_range[0], loc_range[1])
            n_trans = random.randint(trans_range[0], trans_range[1])
            n_prod = random.randint(prod_range[0], prod_range[1])
            sparsity = random.uniform(spars_range[0], spars_range[1])

            # 3. Derive deterministic seed
            current_seed = None
            if master_seed is not None:
                current_seed = master_seed + i

            # 4. Generate instance
            instance = generate_multibatching_problem(
                num_locations=n_loc,
                num_transport_types=n_trans,
                num_products=n_prod,
                demand_sparsity=sparsity,
                seed=current_seed,
                **fixed_kwargs
            )
            dataset.append(instance)
            
            print(f"  [Instance {i+1}/{n_instances}] Config: "
                  f"Loc={n_loc} (from {loc_range}), "
                  f"Prod={n_prod} (from {prod_range}), "
                  f"Trans={n_trans}")

        except Exception as e:
            print(f"  [Error] Failed to generate instance {i+1}: {e}")

    print(f"Generation complete. {len(dataset)} instances created.")
    return dataset

# ==============================================================================
# 2. BENCHMARK ENGINE
# ==============================================================================

def _extract_solution(raw_result: Any):
    """
    Normalizes the result returned by different runners.
    
    Runners may return different data structures depending on the underlying solver 
    (e.g., discrete-optimization ResultStorage, lists of tuples, or raw objects).
    """
    if raw_result is None:
        return None
        
    # Case 1: Standard ResultStorage (from discrete-optimization framework)
    if hasattr(raw_result, "get_best_solution"):
        res = raw_result.get_best_solution()
        if isinstance(res, tuple): # Can return (solution, fitness)
            return res[0]
        return res
    
    # Case 2: List of tuples (e.g., [(solution, fitness), ...])
    if isinstance(raw_result, list) and len(raw_result) > 0:
        last_elem = raw_result[-1] 
        if isinstance(last_elem, tuple):
            return last_elem[0]
        return last_elem
    
    # Case 3: Raw solution object
    return raw_result

def run_benchmark(
    instances: List[MultibatchingProblem], 
    runners: Dict[str, Callable], 
    parameters_cp: ParametersCp,
    time_limit: int,
    **runner_kwargs
) -> dict:
    """
    Executes a set of runners against a list of problem instances and collects KPIs.

    Args:
        instances: List of problems to solve.
        runners: Dictionary mapping display names to runner functions.
        parameters_cp: CP-SAT parameters object (for CP-based runners).
        time_limit: Time limit in seconds for each runner execution.
        **runner_kwargs: Additional keyword arguments passed to runners.

    Returns:
        Dict containing execution metrics for each instance and runner.
    """
    
    benchmark_results = {}
    print(f"\n--- Starting Benchmark with {len(runners)} runners on {len(instances)} instances ---")

    for idx, problem in enumerate(instances):
        instance_id = f"Instance_{idx+1}"
        benchmark_results[instance_id] = {}
        print(f"\n[Processing {instance_id}]")
        
        for runner_name, runner_func in runners.items():
            print(f"  > Running {runner_name}...", end=" ", flush=True)
            start_time = time.time()
            
            try:
                # 1. Execute the imported runner function
                raw_result = runner_func(
                    problem=problem, 
                    p=parameters_cp, 
                    time_limit=time_limit, 
                    **runner_kwargs
                )
                
                # 2. Extract the actual Solution object
                solution = _extract_solution(raw_result)
                elapsed_time = time.time() - start_time
                
                if solution is None:
                    raise ValueError("Runner returned None (no solution found)")

                # 3. Evaluate Solution
                metrics = problem.evaluate(solution)
                total_cost = sum(metrics.values())
                
                # Check Feasibility
                is_feasible = True
                if hasattr(problem, "satisfy"):
                    is_feasible = problem.satisfy(solution)

                # 4. Record Results
                status = "OK" if is_feasible else "INFEASIBLE"
                benchmark_results[instance_id][runner_name] = {
                    "status": status,
                    "time_seconds": round(elapsed_time, 4),
                    "total_cost": round(total_cost, 2),
                    "metrics": metrics, 
                }
                
                print(f"Done in {elapsed_time:.2f}s | {status} | Cost: {total_cost:.2e}")

                
            except Exception as e:
                print(f"FAILED! Error: {e}")
                benchmark_results[instance_id][runner_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "time_seconds": round(time.time() - start_time, 4)
                }
                
    return benchmark_results

def print_comparison_table(results: dict):
    """
    Prints a formatted ASCII table comparing solver performance.
    """
    print("\n" + "="*110)
    print(f"{'INSTANCE':<12} | {'RUNNER':<25} | {'STATUS':<10} | {'TIME (s)':<10} | {'COST':<15} | {'TRANSPORT':<10}")
    print("-" * 110)
    
    for instance_id, runners_res in results.items():
        first = True
        for runner_name, res in runners_res.items():
            inst_label = instance_id if first else "" 
            
            if res.get("status") in ["OK", "INFEASIBLE"]:
                trans_cost = res['metrics'].get('transport', 0)
                print(f"{inst_label:<12} | {runner_name:<25} | {res['status']:<10} | {res['time_seconds']:<10} | {res['total_cost']:<15.2e} | {trans_cost:<10.2e}")
            else:
                print(f"{inst_label:<12} | {runner_name:<25} | {res.get('status', 'ERR'):<10} | {res.get('time_seconds', '-'):<10} | {'ERROR':<15} | {'-':<10}")
            
            first = False
        print("-" * 110)

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Generation Parameters
    LOC_CLUSTERS = [(6, 20), (21, 35), (36, 50)]
    TRANS_CLUSTERS = [(2, 5), (6, 10), (11, 15)]
    PROD_CLUSTERS = [(2, 12), (13, 23), (24, 34)]
    SPARSITY_CLUSTERS = [(0.2, 0.4), (0.6, 0.8)]
    
    # 2. Dataset Generation
    problems = generate_stratified_dataset(
        n_instances=3, 
        location_intervals=LOC_CLUSTERS,
        transport_intervals=TRANS_CLUSTERS,
        product_intervals=PROD_CLUSTERS,
        sparsity_intervals=SPARSITY_CLUSTERS,
        master_seed=42,
        # min_dist=50,
        # max_dist=500,
        # max_tt_cost=100
    )

    # 3. CP-SAT Configuration
    cp_params = ParametersCp.default_cpsat()
    cp_params.nb_process = 16
    
    # Global Time Limit (seconds per runner)
    GLOBAL_TIME_LIMIT = 30 

    # 4. Runners Registration
    # Mapping: Display Name -> Runner Function
    runners_registry = {
        "OpenEvolve (GA)": run_openevolve,
        "NetX + CP-SAT": run_networkx,
        # "Greedy Packing": run_greedy_packing,
        # "Flow PostPro": run_flow_postpro
    }

    # 5. Run Benchmark
    results = run_benchmark(
        instances=problems, 
        runners=runners_registry,
        parameters_cp=cp_params,
        time_limit=GLOBAL_TIME_LIMIT,
        # Additional kwargs for specific runners
        nb_iteration_max=200 
    )

    # 6. Report
    print_comparison_table(results)