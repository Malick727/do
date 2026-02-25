import logging
from pyscipopt import Pricer, SCIP_RESULT
from multibatching.solvers.kh.src.logic import CostCalculator

logger = logging.getLogger(__name__)

class BasePricer(Pricer):
    r"""
    Base class for all Column Generation Pricers.
    
    Mathematical Formulation (Subproblem / Pricing):
    ------------------------------------------------
    We look for a column (pattern $B$) with negative Reduced Cost (Minimization).
    
    $$ RC = \text{FixedCost}_r + \sum_{p \in P} \alpha_p^B (\text{Interest}_p - \mu_{tl,p}) $$
    
    Where:
    * $\mu_{tl,p}$ : Dual value from Link Capacity Constraint.
    * $\alpha_p^B$ : Quantity of product $p$ in pattern $B$.
    * $\text{Interest}_p$ : Holding cost for product $p$ during transit.
    
    Objective:
    $$ \min RC \quad \text{s.t.} \quad \min RC < -\epsilon $$
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a pricer implementation (e.g. @register('scip'))."""
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get_pricer_class(cls, name):
        return cls._registry.get(name)

    def __init__(self, constraints, products, links, routes, settings, mode):
        super().__init__()
        self.constraints = constraints
        self.products = products
        self.links = links
        self.routes = routes
        self.settings = settings
        self.mode = mode
        self.transformed_constraints = {}
        self.pattern_registry = {}

    def pricerinit(self):
        """Initialize transformed constraints map for fast dual retrieval."""
        for key, cons in self.constraints.items():
            if cons:
                self.transformed_constraints[key] = self.model.getTransformedCons(cons)

    def get_duals(self):
        """
        Retrieves dual values from the Master Problem.
        Returns:
            dict: Key=(TransportLink, Product), Value=Dual Value (float)
        """
        duals = {}
        for link in self.links:
            for product in self.products:
                key = (link, product)
                val = 0.0
                if key in self.transformed_constraints:
                    try:
                        val = self.model.getDualsolLinear(self.transformed_constraints[key])
                    except Exception:
                        val = 0.0
                duals[key] = val
        return duals

    def prepare_subproblem_data(self, route):
        """
        Helper: extracts all necessary metrics for a route and filters valid products.
        Returns:
            tuple: (capacity, duration, fixed_cost, valid_products_list)
        """
        capacity = route.transport_resource.capacity
        duration = CostCalculator.calculate_route_duration(route)
        fixed_cost = CostCalculator.calculate_transport_cost(route, self.settings)
        
        valid_products = [
            p for p in self.products
            if route.transport_resource.name in p.valid_transport_resources
        ]
        
        return capacity, duration, fixed_cost, valid_products

    def pricerredcost(self):
        """Standardized SCIP callback."""
        duals = self.get_duals()
        new_patterns = self.find_patterns(duals)

        if not new_patterns:
            return {"result": SCIP_RESULT.SUCCESS}

        columns_added = 0
        for data in new_patterns:
            self.add_new_column(
                data['route'], 
                data['pattern'], 
                data['duration'], 
                data['fixed_cost']
            )
            columns_added += 1

        logger.info(f"[{self.__class__.__name__}] Added {columns_added} columns.")
        return {"result": SCIP_RESULT.SUCCESS}

    def find_patterns(self, duals):
        """ABSTRACT METHOD."""
        raise NotImplementedError("Subclasses must implement find_patterns")

    def add_new_column(self, route, pattern, duration, fixed_cost):
        """Adds a new variable (column) to the Master Problem."""
        total_value = sum(p.value * qty for p, qty in pattern.items())
        interest_cost = CostCalculator.calculate_interest(
            total_value, duration, self.settings
        )
        
        total_obj_cost = fixed_cost + interest_cost
        
        pat_id = len(self.pattern_registry)
        var_name = (f"pat_{repr(route.transport_link)}_"
                    f"{route.transport_resource.name}_{pat_id}")
        
        var_vtype = "I" if "ip" in self.mode else "C"
        
        new_var = self.model.addVar(
            name=var_name,
            vtype=var_vtype, 
            obj=total_obj_cost,
            pricedVar=True
        )

        self.pattern_registry[var_name] = {
            "variable": new_var,
            "route": route,
            "pattern": pattern, 
            "cost": total_obj_cost
        }

        for product, qty in pattern.items():
            if qty > 0:
                key = (route.transport_link, product)
                if key in self.transformed_constraints:
                    self.model.addConsCoeff(self.transformed_constraints[key], new_var, qty)