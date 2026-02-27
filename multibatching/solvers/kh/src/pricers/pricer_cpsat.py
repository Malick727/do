import multiprocessing
import logging
from ortools.sat.python import cp_model
from multibatching.solvers.kh.src.logic import CostCalculator
from multibatching.solvers.kh.src.pricers.base_pricer import BasePricer
from functools import partial

logger = logging.getLogger(__name__)


def solve_subproblem_worker(task_data, time_limit=2):
    """
    Worker to solve the Knapsack Subproblem using OR-Tools CP-SAT.
    """
    route = task_data['route']
    valid_products = task_data['valid_products']
    duals = task_data['duals']
    capacity = task_data['capacity']
    duration = task_data['duration']
    fixed_cost = task_data['fixed_cost']
    settings = task_data['settings']

    if not valid_products:
        return None

    model = cp_model.CpModel()

    # Scaling factor for float coefficients (CP-SAT is an integer solver)
    SCALING = 1

    # Variables: q_p (Quantity of product p)
    vars_dict = {}
    for p in valid_products:
        # Define upper bound based on capacity to keep search space tight
        u_bound = int(capacity // p.size) if p.size > 0 else 1000
        vars_dict[p] = model.NewIntVar(0, u_bound, f"q_{p.id}")

    # Constraint: Capacity
    model.Add(
        sum(vars_dict[p] * int(p.size) for p in valid_products) <= int(capacity)
    )

    # Objective: Minimize Reduced Cost
    # RC = fixed_cost + sum( q_p * (interest_per_unit - dual_p) )
    # Since CP-SAT is integer only, we scale the float costs.
    scaled_obj = []

    for p in valid_products:
        interest_per_unit = CostCalculator.calculate_interest(p.value, duration, settings)
        mu_val = duals.get((route.transport_link, p), 0.0)

        # Coefficient: (interest - dual)
        coeff = int((interest_per_unit - mu_val) * SCALING)
        scaled_obj.append(vars_dict[p] * coeff)

    # Add the constant fixed_cost (scaled)
    scaled_fixed_cost = int(fixed_cost * SCALING)
    model.Minimize(sum(scaled_obj) + scaled_fixed_cost)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    # Enable internal parallelization if desired,
    # but since we are already using multiprocessing, 1 thread per worker is safer.
    solver.parameters.num_search_workers = 1

    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        min_rc = solver.ObjectiveValue() / SCALING

        if min_rc < -settings.epsilon:
            pattern = {p.id: solver.Value(vars_dict[p]) for p in valid_products}
            return {
                "route": route,
                "pattern_data": pattern,
                "duration": duration,
                "fixed_cost": fixed_cost,
                "reduced_cost": min_rc
            }

    return None


@BasePricer.register("cpsat")  # Updated tag
class CpSatKnapsackPricer(BasePricer):
    def find_patterns(self, duals):
        if not self.run_initial and self.options.get("add_single_batch", True):
            self.run_initial = True
            return self.generate_initial_patterns()
        tasks = []
        for route in self.routes:
            cap, dur, f_cost, valid_prods = self.prepare_subproblem_data(route)

            if not valid_prods:
                continue

            tasks.append({
                'route': route,
                'valid_products': valid_prods,
                'capacity': cap,
                'duration': dur,
                'fixed_cost': f_cost,
                'duals': duals,
                'settings': self.settings
            })

        # Now you can safely use all CPU cores
        num_workers = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(
                partial(solve_subproblem_worker, time_limit=self.options.get("time_limit", 2)),
                tasks
            )

        formatted_results = []
        prod_map = {p.id: p for p in self.products}
        best_rc = 0.0

        for res in results:
            if res and res['reduced_cost'] < -self.settings.epsilon:
                if res['reduced_cost'] < best_rc:
                    best_rc = res['reduced_cost']

                pattern_obj = {prod_map[p_id]: qty for p_id, qty in res['pattern_data'].items()}
                formatted_results.append({
                    'route': res['route'],
                    'pattern': pattern_obj,
                    'duration': res['duration'],
                    'fixed_cost': res['fixed_cost']
                })

        if best_rc < 0:
            logger.info(f"Best Reduced Cost: {best_rc:.4f}")

        return formatted_results